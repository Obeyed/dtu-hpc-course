# High Performance Computing
## Special Course
### The Technical University of Denmark - January, 2016
This repository contains our documentation for a special course in high performance computing at The Technical University of Denmark.
We have a folder containing our report, and one folder containing the code.

# Smart ideas

- Nvidia NSight
- testing these things http://stackoverflow.com/questions/11816786/why-bother-to-know-about-cuda-warps
- setting up guide http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3wZxUxpLy
- https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf

# Warp testing
Num_threads Time
- 32x10:    5.321
- 32x10-1:  5.336
- 32x10-16: 5.600
- 32x10-31: 5.892
- 32x9:     5.561
- 32x9-1:   5.585
- 32x9-16:  5.892
- 32x9-31:  6.236

# Overhead testing
Num_threads=1<<10
Model       Time
- Base:     5.437
- Init:     5.437
- global read:  (coalescled)
- globla read: (not coalescled)
- shared write:
- global write:
