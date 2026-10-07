▶ python3 tools/plot_compare.py --ref out_ref --cmp tf32=out_tf32 --all-steps
plot_compare: 42 common frame(s), steps 0..20000
plot_compare: step      0  t=0.000000  tf32 Linf=0.000e+00 RMS=0.000e+00
plot_compare: step    200  t=0.020000  tf32 Linf=1.348e-04 RMS=1.245e-05
plot_compare: step    512  t=0.051200  tf32 Linf=1.653e-04 RMS=1.260e-05
plot_compare: step   1024  t=0.102400  tf32 Linf=1.921e-04 RMS=1.469e-05
plot_compare: step   1536  t=0.153600  tf32 Linf=3.115e-04 RMS=2.048e-05
plot_compare: step   2048  t=0.204800  tf32 Linf=4.530e-04 RMS=2.829e-05
plot_compare: step   2560  t=0.256000  tf32 Linf=3.158e-04 RMS=2.019e-05
plot_compare: step   3072  t=0.307200  tf32 Linf=2.631e-04 RMS=1.764e-05
plot_compare: step   3584  t=0.358400  tf32 Linf=2.370e-04 RMS=1.612e-05
plot_compare: step   4096  t=0.409600  tf32 Linf=1.922e-04 RMS=1.510e-05
plot_compare: step   4608  t=0.460800  tf32 Linf=1.557e-04 RMS=1.502e-05
plot_compare: step   5120  t=0.512000  tf32 Linf=1.349e-04 RMS=1.645e-05
plot_compare: step   5632  t=0.563200  tf32 Linf=1.101e-04 RMS=1.750e-05
plot_compare: step   6144  t=0.614400  tf32 Linf=8.839e-05 RMS=1.835e-05
plot_compare: step   6656  t=0.665600  tf32 Linf=1.110e-04 RMS=1.815e-05
plot_compare: step   7168  t=0.716800  tf32 Linf=1.808e-04 RMS=1.974e-05
plot_compare: step   7680  t=0.768000  tf32 Linf=2.323e-04 RMS=2.594e-05
plot_compare: step   8192  t=0.819200  tf32 Linf=2.074e-04 RMS=2.732e-05
plot_compare: step   8704  t=0.870400  tf32 Linf=1.881e-04 RMS=2.921e-05
plot_compare: step   9216  t=0.921600  tf32 Linf=1.670e-04 RMS=3.127e-05
plot_compare: step   9728  t=0.972800  tf32 Linf=1.418e-04 RMS=2.856e-05
plot_compare: step  10240  t=1.024000  tf32 Linf=1.073e-04 RMS=2.764e-05
plot_compare: step  10752  t=1.075200  tf32 Linf=7.315e-05 RMS=2.238e-05
plot_compare: step  11264  t=1.126400  tf32 Linf=6.548e-05 RMS=2.177e-05
plot_compare: step  11776  t=1.177600  tf32 Linf=7.934e-05 RMS=2.275e-05
plot_compare: step  12288  t=1.228800  tf32 Linf=1.061e-04 RMS=2.003e-05
plot_compare: step  12800  t=1.280000  tf32 Linf=1.234e-04 RMS=2.172e-05
plot_compare: step  13312  t=1.331200  tf32 Linf=1.237e-04 RMS=2.278e-05
plot_compare: step  13824  t=1.382400  tf32 Linf=1.534e-04 RMS=2.682e-05
plot_compare: step  14336  t=1.433600  tf32 Linf=8.590e-05 RMS=1.605e-05
plot_compare: step  14848  t=1.484800  tf32 Linf=5.734e-05 RMS=1.233e-05
plot_compare: step  15360  t=1.536000  tf32 Linf=5.785e-05 RMS=1.124e-05
plot_compare: step  15872  t=1.587200  tf32 Linf=5.164e-05 RMS=1.311e-05
plot_compare: step  16384  t=1.638400  tf32 Linf=6.893e-05 RMS=1.335e-05
plot_compare: step  16896  t=1.689600  tf32 Linf=9.018e-05 RMS=1.462e-05
plot_compare: step  17408  t=1.740800  tf32 Linf=7.202e-05 RMS=1.358e-05
plot_compare: step  17920  t=1.792000  tf32 Linf=1.194e-04 RMS=1.280e-05
plot_compare: step  18432  t=1.843200  tf32 Linf=5.397e-05 RMS=1.230e-05
plot_compare: step  18944  t=1.894400  tf32 Linf=7.898e-04 RMS=1.381e-05
plot_compare: step  19456  t=1.945600  tf32 Linf=1.029e-04 RMS=1.349e-05
plot_compare: step  19968  t=1.996800  tf32 Linf=6.035e-05 RMS=1.421e-05
plot_compare: step  20000  t=2.000000  tf32 Linf=1.017e-04 RMS=1.434e-05
plot_compare: wrote 42 frame(s) into out_compare/
plot_compare: wrote out_compare/compare.gif (42 frames)
plot_compare: wrote out_compare/error_vs_time.csv
plot_compare: wrote out_compare/error_vs_time.png
plot_compare: tf32     area drift  ref=2.8368%  tf32=2.8584%  delta=0.0216 pp
