import math 

input_scale = (1920, 1280)
ratio_range = (0.8, 1.2)


ratio_diff = ratio_range[1] - ratio_range[0]

lower = (int(input_scale[0] * ratio_range[0]), int(input_scale[1] * ratio_range[0]))
upper = (int(input_scale[0] * ratio_range[1]), int(input_scale[1] * ratio_range[1]))

diff = (int(input_scale[0] * ratio_diff), int(input_scale[1] * ratio_diff))
bin = (diff[0] / 6, diff[1] / 6)
res = []
for i in range(7):
    if i == 0 :
        res.append(lower)
    elif i == 6 :
        res.append(upper)
    else :
        res.append((int(res[-1][0] + bin[0]), int(res[-1][1] + bin[1])))

print(res)


