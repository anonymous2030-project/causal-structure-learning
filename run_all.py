"""Run all three experiments end-to-end and write figures + tables to results/."""
import time
import exp1_estimator, exp2_energy, exp3_latency

if __name__ == "__main__":
    t0 = time.time()
    print("\n=== Experiment 1: structure recovery + detection specificity ===")
    exp1_estimator.run()
    print("\n=== Experiment 2: energy efficiency under jamming ===")
    exp2_energy.run_exp()
    print("\n=== Experiment 3: real-time latency & scaling ===")
    exp3_latency.main()
    print("\nAll experiments done in %.1f s. See results/." % (time.time() - t0))