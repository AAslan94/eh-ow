import numpy as np
import cma
import sys
import multiprocessing

from libow8 import sensor_net
import owutils as ut
from designs_diag import designs, align_receiver_to_transmitter
from panel_ow import Panel

KEY = 'B2'
K   = 40   # fixed number of perturbations — worst-case taken as score

# ==========================================
# ROBUST SENSOR EVALUATOR
# ==========================================

class RobustSensorEvaluator:
    def __init__(self, r_sens, nS_sens, params_d, params_amb):
        self.r_sens     = r_sens
        self.nS_sens    = nS_sens
        self.params_d   = params_d
        self.params_amb = params_amb
        self.h_ww  = None
        self.h_amb = None
        self.max_rad = 0.087266  # 5 degrees

        # Pre-generate all 40 perturbations once with a fixed seed.
        # Using a fixed seed means the objective is fully deterministic —
        # the same candidate always gets the same score.
        rng = np.random.RandomState(42)

        cos_max = np.cos(self.max_rad)
        fixed_cos            = rng.uniform(cos_max, 1.0, size=K)
        self.fixed_angles    = np.arccos(fixed_cos)

        # Azimuthal axes: uniform random unit vectors, projected perpendicular
        # to n_nom at evaluation time, covering the full spherical cap.
        axis_raw             = rng.randn(K, 3)
        norms                = np.linalg.norm(axis_raw, axis=1, keepdims=True)
        self.fixed_axes      = axis_raw / (norms + 1e-12)

    def _perturb(self, nR_nom, angle, axis):
        """
        Rodrigues rotation of nR_nom by angle around axis.
        axis is made perpendicular to nR_nom so the third Rodrigues term vanishes.
        """
        axis  = axis.reshape(1, 3)
        proj  = np.sum(axis * nR_nom, axis=1, keepdims=True)
        ortho = axis - proj * nR_nom
        norms = np.linalg.norm(ortho, axis=1, keepdims=True)
        ortho = np.divide(ortho, norms,
                          out=np.zeros_like(ortho), where=norms > 1e-12)
        return nR_nom * np.cos(angle) + np.cross(ortho, nR_nom) * np.sin(angle)

    def evaluate_batch(self, x_norm_batch):
        """
        Evaluates all particles against all 40 perturbations.
        Returns the worst-case cost per particle: shape (n_particles,).
        """
        n_particles = x_norm_batch.shape[0]

        theta = x_norm_batch[:, 0] * (np.pi / 2)
        phi   = (x_norm_batch[:, 1] % 1.0) * (2 * np.pi)
        area  = 0.0001 + x_norm_batch[:, 2] * (0.015 - 0.0001)

        r_sensor = np.tile(self.r_sens, (n_particles, 1))
        nR_nom   = ut.spher_to_cart_ar(1, theta, phi).T

        fitness_samples = np.zeros((n_particles, K))

        for k in range(K):
            nR = self._perturb(nR_nom, self.fixed_angles[k], self.fixed_axes[k])

            for p in (self.params_amb, self.params_d):
                p['r_sensor']  = r_sensor
                p['nR_sensor'] = nR
                p['A_sensor']  = area
                p['nS_sensor'] = self.nS_sens

            l_amb = sensor_net(**self.params_amb)
            l_amb.calch(h_ww=self.h_amb)
            l_amb.light_sim()
            if self.h_amb is None:
                self.h_amb = l_amb.h_ww

            net_d = sensor_net(**self.params_d)
            net_d.calch(h_ww=self.h_ww)
            net_d.light_sim()
            if self.h_ww is None:
                self.h_ww = net_d.h_ww

            net_d.calc_noise()
            net_d.calc_rq()

            p_all = (
                np.sum(np.sum(l_amb.Pin_sm_diff, axis=0), axis=1)
                + np.sum(l_amb.Pin_sm, axis=0)
                + l_amb.Pin_sa
                + 0.5 * net_d.Pin_sm_tot.flatten()
            )

            G_ac  = net_d.Pin_sm_tot.flatten() / area
            G_all = p_all / area

            snr_dB   = np.zeros(n_particles)
            e_day    = np.zeros(n_particles)
            bitrates = np.zeros(n_particles)
            freq     = np.linspace(100, 20000, 50)

            for i in range(n_particles):
                p = Panel(
                    area[i] * 1e4,
                    rs=1, rsh=1000, n=1.6,
                    voc=0.64, isc=35e-3,
                    G=G_all[i], Gac=G_ac[i]
                )
                p.run(False)

                ind = int(p.ind)
                p.calc_capacitance()
                p.set_circuit(Rc=10, Lo=100e-6, Co=1000e-6)
                p.find_bw()
                p.tf(freq)
                p.all_thermal_noise(freq)
                p.shot_noise(freq)
                p.vp2p(freq)

                sig   = p.vac[ind] + 1e-20
                noise = 4 * (p.th_noise[ind] + p.sh_noise[ind]) + 1e-20
                snr_dB[i]    = 10 * np.log10((sig ** 2) / noise)
                e_day[i]     = p.Pmax * 3600 * 4 * 0.8
                bitrates[i]  = p.BW[ind] * 0.4

            net_d.calc_tbattery(br=bitrates)

            V          = 3.3
            cycle_p    = np.array([c.calc_cycle_consumption() for c in net_d.cycles]) * V
            cycles_day = (3600 * 24) / net_d.Tcycle
            e_day_c    = cycles_day * cycle_p

            penalty = np.zeros(n_particles)
            mask_snr = snr_dB < 8.5
            penalty[mask_snr] += ((8.5 - snr_dB[mask_snr]) ) * 1e4
            mask_eng = e_day < e_day_c
            penalty[mask_eng] += ((e_day_c[mask_eng] - e_day[mask_eng])) * 1e4

            fitness_samples[:, k] = area + penalty

        # Minimax: worst case over all 40 perturbations
        return np.max(fitness_samples, axis=1)


# ==========================================
# WORKER FUNCTION
# ==========================================

def run_sensor_optimization(args):
    idx, r_val, nS_val, p_d_copy, p_amb_copy = args

    evaluator = RobustSensorEvaluator(r_val, nS_val, p_d_copy, p_amb_copy)

    es = cma.CMAEvolutionStrategy(
        3 * [0.5], 0.4,
        {
            'bounds':  [[0, None, 0], [1, None, 1]],
            'popsize': 50,
            'verbose': -9,
            'maxiter': 80
        }
    )

    gen = 0
    log_every = 10  # print every 10 generations

    while not es.stop():
        X = np.array(es.ask())
        F = evaluator.evaluate_batch(X)
        es.tell(X.tolist(), F.tolist())
        gen += 1

        if gen % log_every == 0:
            best = es.result.fbest
            print(
                f"  [Sensor {idx+1:02d}] Gen {gen:4d}/{80} | "
                f"Best={best:.5f} | sigma={es.sigma:.4f}",
                flush=True
            )

    best  = es.result.xbest
    theta = best[0] * (np.pi / 2)
    phi   = (best[1] % 1.0) * (2 * np.pi)
    area  = 0.0001 + best[2] * (0.015 - 0.0001)

    return {
        'id':    idx,
        'pos':   r_val,
        'theta': theta,
        'phi':   phi,
        'area':  area,
        'cost':  es.result.fbest
    }


# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    r_sen_list  = designs[KEY]['r_sensor']
    nS_sen_list = np.round(
        align_receiver_to_transmitter(r_sen_list, np.array([4, 3, 2.8])), 2
    )
    N_sensors = r_sen_list.shape[0]

    base_params_d   = designs[KEY].copy()
    base_params_amb = designs['B2'].copy()
    base_params_amb['r_master']  = base_params_d['r_lights']
    base_params_amb['PT_master'] = base_params_d['PT_lights']

    tasks = [
        (i, r_sen_list[i], nS_sen_list[i],
         base_params_d.copy(), base_params_amb.copy())
        for i in range(N_sensors)
    ]

    print(f"Starting MINIMAX ROBUST CMA-ES for {N_sensors} sensors (KEY={KEY})...")
    print(f"Using {K} fixed perturbations — score = worst-case cost.")
    print("Running on 8 processors...")

    final_results_list = []

    with multiprocessing.Pool(processes=8) as pool:
        for res in pool.imap_unordered(run_sensor_optimization, tasks):
            print(
                f"Sensor {res['id']+1:02d} -> "
                f"Theta={np.degrees(res['theta']):5.1f}° | "
                f"Phi={np.degrees(res['phi']):5.1f}° | "
                f"Area={res['area']*1e4:5.2f} cm² | "
                f"Cost={res['cost']:.4f}"
            )
            sys.stdout.flush()
            final_results_list.append(res)

    final_results_list.sort(key=lambda x: x['id'])

    data_to_save = np.array([
        np.hstack((d['pos'], [d['theta'], d['phi'], d['area'], d['cost']]))
        for d in final_results_list
    ])

    filename = f'robust_minimax_results_{KEY}.npy'
    np.save(filename, data_to_save)

    print("-" * 60)
    print(f"Results saved to '{filename}'.")
    print(f"Array shape: {data_to_save.shape}")
    print("Columns: [Pos_x, Pos_y, Pos_z, Theta, Phi, Area, Cost]")
