# ActSTD

The official implementation of:  [Activity Trajectory Generation via Modeling Spatiotemporal Dynamics (KDD '22)](https://dl.acm.org/doi/abs/10.1145/3534678.3542671).

## Installation

### Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1

### Dependencies
1. Install PyTorch 1.7.1 with the correct CUDA version.
2. Use the ``pip install -r requirements. txt`` command to install all of the Python modules and packages used in this project.

run `python src/setup.py build_ext --inplace` to create the shared object file in the current directory.


## Model Training
`cd src`

Use the following command to train ActSTD on `Mobile` dataset with different CNF models: 

`python app.py --data Mobile --model attncnf --tpp neural  --l2_attn --ode_method 'scipy_solver' --ode_solver 'RK45' --cuda_id 0 --tpp_style 'gru'  --weekhour`

`python app.py --data Mobile --model jumpcnf --tpp neural --solve_reverse --ode_method 'scipy_solver' --ode_solver 'RK45' --cuda_id 0 --tpp_style 'gru' --weekhour`

Use the following command to train ActSTD on `Foursquare` dataset with different CNF models: 

`python app.py --data Foursquare --model attncnf --tpp neural  --l2_attn --ode_method 'scipy_solver' --ode_solver 'RK45' --cuda_id 0 --tpp_style 'gru'  --weekhour`

`python app.py --data Foursquare --model jumpcnf --tpp neural --solve_reverse --ode_method 'scipy_solver' --ode_solver 'RK45' --cuda_id 0 --tpp_style 'gru' --weekhour`

## More Related Works
- [Learning to Simulate Daily Activities via Modeling Dynamic Human Needs (WWW'23)](https://github.com/tsinghua-fib-lab/Activity-Simulation-SAND)

## Citation
If you find this repository useful in your research, please consider citing the following paper:

```
@inproceedings{yuan2022activity,
  title={Activity Trajectory Generation via Modeling Spatiotemporal Dynamics},
  author={Yuan, Yuan and Ding, Jingtao and Wang, Huandong and Jin, Depeng and Li, Yong},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4752--4762},
  year={2022}
}
```

## Note
The implemention is based on [Neural STPP](https://github.com/facebookresearch/neural_stpp).
