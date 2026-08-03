# Curvature_aware_dynamic_precision_for_PINN
- The repository for paper "Curvature-Aware Dynamic Precision Approach for Physics-Informed Neural Networks".

@misc{shao2026curvatureawaredynamicprecisionapproach,
      title={Curvature-aware dynamic precision approach for physics-informed neural networks}, 
      author={Yingjie Shao and Ioannis N. Athanasiadis and George van Voorn and Taniya Kapoor},
      year={2026},
      eprint={2606.04736},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.04736}, 
}

## Repository Structure

```text
.
├── code/
│   ├── percision_dynamic_switching.py #main script for curvature aware dynamic percision with 4 1D PDE failture mode benchmarks, 2 PDE systems and one 50D heat equation. This script also include implementation for adaptive sampling and weighting method for PINN. Detail comment is include in corresponding functions.
│   ├── convection_full_slq_hessian_esd_update.py # Appendix Empirical validation of the curvature proxy
│   ├── plot_and_result_summary.py # script for generate table and figures for paper
│   └── ssbroyden2_torch.py #The file that implement ssbroyden2 optimiser with pytorch, which is under BSD 3-Clause License
├── figure/
│   └── generated paper figures
├── LICENSE
├── THIRD_PARTY_NOTICES
├── requirements.txt
└── README.md
```

## License

Except where otherwise noted, this repository is licensed under the
[MIT License](LICENSE).

The file [`code/ssbroyden2_torch.py`](code/ssbroyden2_torch.py) is licensed
under the BSD 3-Clause License. Its complete copyright notices, license
conditions, and disclaimers are provided in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
