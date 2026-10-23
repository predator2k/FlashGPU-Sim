# YASAI_FUNC_1SM

The SM120_RTX5090 configuration with one SM (`-gpgpu_n_clusters 1`) for the pure functional mode
(`PTX_SIM_MODE_FUNC=1`). The functional mode executes one CTA at a time and does not use the
per-SM timing structures, but FlashGPU-Sim still allocates them at start-up: with 170 SMs a
process holds about 3.9 GB before running anything, with one SM about 0.5 GB. The memory
partition count stays at 16 because the IPOLY L2 channel indexing needs 16 to 1024 channels.
Functional results (instructions, addresses) do not depend on the SM count.
