#cython: language_level=3, wraparound=False, boundscheck=False, cdivision=True
from libc.math cimport fabs, sqrt, pow, atan, sin, cos, tan, exp
from libc.stdlib cimport rand, srand
import time
cimport cython
from cpython cimport array
import array
import numpy as np
cdef extern from "stdlib.h":
    int RAND_MAX

# boat boundaries 
vb = np.array([[0.0, 10.0, 0.0, -10.0],[12.9, 0.0, -11.0, 0.0]], dtype=np.float32, order='C').T
pb = np.array([[0.0, 5.0, 0.0, -5.0],[12.9, 0.0, -11.0, 0.0]], dtype=np.float32, order='C').T
vb = vb.copy(order='C')
pb = pb.copy(order='C')
cdef float [:,::1] virtual_boundary = vb
cdef float [:,::1] physical_boundary = pb

# reward scaling
cdef float  penalty = 0.05
cdef float  game_penalty = 0.05
cdef float  start_penalty = 0.05
cdef float  collision_penalty = 0.01
cdef float  final_reward_scale = 1

# polar velocity
data = np.loadtxt("polar.csv", delimiter=",", dtype=np.float32)
cdef float [:,::1] polar = data
cdef float min_tws = 3.6
cdef float max_tws = 10.8
cdef float tws_step = 1.029
cdef int n_tws = data.shape[0]
cdef float  v_min =0.5
cdef float  vmg_cwa = 0.82

# coefficients for acceleration
cdef float ac1 = 0.035
cdef float ac2 = -0.009
cdef float ac3 = 0.5
cdef float ac4 = 0.027
cdef float ac5 = 6.77
cdef float ac6 = 3.68
cdef float ac7 = -0.0003
cdef float ac8 = 45.8
cdef float pi = 3.14159
cdef float pi_2 = 1.5708
cdef float turn_rate_limit = 0.6981317007977318

# episode end criteria
cdef float t_after_start = 20
cdef float dmg_after_start = 50

# start line
cdef float line_len = 200
cdef float line_skew = 0
cdef float line_len_variability = 100
cdef float line_skew_variability = 0.5

# start box
cdef float box_width = 1300 / 2
cdef float box_depth = 1300

# indexs for variable positions in the state vector
cdef int idx_b1x = 0
cdef int idx_b1y = 1
cdef int idx_b1v = 2
cdef int idx_b1cwa = 3
cdef int idx_b1tr = 4
cdef int idx_b1ent = 5
cdef int idx_b1start = 6
cdef int idx_b2x = 7
cdef int idx_b2y = 8
cdef int idx_b2v = 9
cdef int idx_b2cwa = 10
cdef int idx_b2tr = 11
cdef int idx_b2ent = 12
cdef int idx_b2start = 13
cdef int idx_t = 14
cdef int idx_prt_x = 15
cdef int idx_prt_y = 16
cdef int idx_stb_x = 17
cdef int idx_stb_y = 18
cdef int idx_row = 19
cdef int idx_row_1 = 20
cdef int idx_row_2 = 21
cdef int idx_tws = 22

################################################################
cdef float rand_float():
    cdef float  f = <float>rand() / <float>RAND_MAX - 0.5
    return f

cdef float rand_float_0_1():
    cdef float  f = <float>rand() / <float>RAND_MAX
    return f

################################################################
cdef float limit_pi(float val):
    if val > pi:
        val -= 2 * pi
    if val < -pi:
        val += 2 * pi
    return val

################################################################
cdef float deg_to_rad(float deg):
    return deg * pi / 180.0

################################################################
cpdef float calc_polar_v(float tws, float cwa):
    cdef float v, v_low, v_high, t_low, t_high, r
    cdef int i, j
    
    # keep values between limits
    if cwa < 0:
        cwa *= -1
    if tws < min_tws:
        tws = min_tws
    if tws > max_tws:
        tws = max_tws
    
    # get the polar velocities for the tws in the table either side of the actual tws
    i = <int>((tws - min_tws) / tws_step)
    if i >= n_tws - 1:
        i = n_tws - 2
    j = <int>(cwa * 18 / pi)
    if j > 17:
        j = 17
    
    r = (cwa - j * pi / 18) / (pi / 18)
    v_low = polar[i,j] * (1 - r) + polar[i,j+1] * r
    v_high = polar[i+1,j] * (1 - r) + polar[i+1,j+1] * r
    
    # interpolate between the polar velocities at the two wind speeds
    r = (tws - (min_tws + i * tws_step)) / tws_step
    v = v_low * (1 - r) + v_high * r
    
    if v < v_min:
        v = v_min
    
    return v

################################################################
cdef float calc_acc(float tws, float cwa, float v, float turn_rate, float displacement_action):
    cdef float  v_polar, acc, delta_bsp, delta_cwa, b, c, x

    # calc polar velocity
    v_polar = calc_polar_v(tws, cwa)
    if displacement_action > 0.5:
        v_polar *= 0.7
    
    # calculate acceleration based on the polar speed
    delta_bsp = v_polar - v
    if delta_bsp > 10:
        delta_bsp = 10
    delta_cwa = fabs(fabs(cwa) - pi_2)
    acc = ac1 * tws * delta_bsp
    acc += ac2 * pow(delta_bsp, 2)
    acc *= (1.0 - ac3 * delta_cwa)
    if v < ac5 + ac6 and v > ac5 - ac6:
        acc += ac4 * (pow(v - ac5, 2) - pow(ac6, 2))
    
    # adjust acceleration from the turn rate
    b = 14.4 * (fabs(cwa) - 1)
    c = -ac7 * pow(b, 2)
    x = ac8 * turn_rate + b
    if cwa < 0:
        x = -ac8 * turn_rate + b
    acc += ac7 * pow(x, 2) + c

    # limit acc to -2m/s^2
    if acc < -2:
        acc = -2

    return acc

################################################################
cdef float calc_turn_rate(float v, float cwa, float prev_tr, float action, float dt):
    cdef float turn_rate, max_turn_rate, max_delta_turn_rate, turn_angle

    # set the turn angle from the action
    turn_angle = action / 5.0
    
    # set the desired turn rate based on turn angle
    turn_rate = fabs(turn_angle) / dt

    # calc limit based on speed and absolute limits
    max_turn_rate = deg_to_rad(v * 2)
    if deg_to_rad(5.0) > max_turn_rate:
        max_turn_rate = deg_to_rad(5.0)
    if max_turn_rate > deg_to_rad(25.0):
        max_turn_rate = deg_to_rad(25.0)

    # calc limit of change in turn rate
    max_delta_turn_rate = max_turn_rate * dt

    # limit turn rate
    if turn_rate > max_turn_rate:
        turn_rate = max_turn_rate
    if turn_angle < 0:
        turn_rate *= -1
    if turn_rate - prev_tr > max_delta_turn_rate:
        turn_rate = prev_tr + max_delta_turn_rate
    if turn_rate - prev_tr < -max_delta_turn_rate:
        turn_rate = prev_tr - max_delta_turn_rate

    return turn_rate

################################################################
cdef void get_vertex_position(float x, float y, float cwa, float vertex, float[:,::1] boundary, float[::1] v_loc):
    cdef int vertex_i = <int> vertex
    v_loc[0] = x + boundary[vertex_i, 0] * cos(cwa) + boundary[vertex_i, 1] * sin(cwa)
    v_loc[1] = y + boundary[vertex_i, 1] * cos(cwa) - boundary[vertex_i, 0] * sin(cwa)
    return

################################################################
def get_bow_location(float[::1] state, b1):
    cdef float x, y
    if b1:
        x = state[idx_b1x] + virtual_boundary[0, 1] * sin(state[idx_b1cwa])
        y = state[idx_b1y] + virtual_boundary[0, 1] * cos(state[idx_b1cwa])
    else:
        x = state[idx_b2x] + virtual_boundary[0, 1] * sin(state[idx_b2cwa])
        y = state[idx_b2y] + virtual_boundary[0, 1] * cos(state[idx_b2cwa])
    return float(x), float(y)

################################################################
cdef int calc_row(float[::1] state):
    # right of way, 1 = boat 1 has right of way, -1 = boat 2 has right of way, 0 = neither boat has right of way
    cdef float x, y, s1x, s2x, b1y, b2y, m
    cdef int flag
    flag = 0

    cdef array.array s1_ = array.array('f', [0, 0])
    cdef array.array s2_ = array.array('f', [0, 0])
    cdef float[::1] s1 = s1_
    cdef float[::1] s2 = s2_

    ################################################################
    # check if they are on different tacks
    if state[idx_b1cwa] * state[idx_b2cwa] < 0:
        # starboard has right of way over port
        if state[idx_b1cwa] > 0:
            return -1
        else:
            return 1
    else:
        # they are on the same tack
        ################################################################
        # check if one boat going upwind and the other downwind
        if fabs(state[idx_b1cwa]) < pi / 2 and fabs(state[idx_b2cwa]) > pi / 2:
            return 1
        if fabs(state[idx_b1cwa]) > pi / 2 and fabs(state[idx_b2cwa]) < pi / 2:
            return -1

        ################################################################
        # they are both going upwind or both going downwind

        # get the intersection point of the lines going through the boats current travel
        x = line_intersection_x(state[idx_b1x], state[idx_b1y], state[idx_b1cwa], state[idx_b2x], state[idx_b2y], state[idx_b2cwa])

        # is the intersection point aft of both bows
        get_vertex_position(state[idx_b1x], state[idx_b1y], state[idx_b1cwa], 1, virtual_boundary, s1)
        get_vertex_position(state[idx_b2x], state[idx_b2y], state[idx_b2cwa], 1, virtual_boundary, s2)
        s1x = s1[0]
        s2x = s2[0]

        if state[idx_b1cwa] > 0:
            if x < s1x and x < s2x:
                flag = 1
        else:
            if x > s1x and x > s2x:
                flag = 1

        # is the intersection point forward of both sterns
        if flag == 0:
            get_vertex_position(state[idx_b1x], state[idx_b1y], state[idx_b1cwa], 3, virtual_boundary, s1)
            get_vertex_position(state[idx_b2x], state[idx_b2y], state[idx_b2cwa], 3, virtual_boundary, s2)
            s1x = s1[0]
            s2x = s2[0]
            if state[idx_b1cwa] > 0:
                if x > s1x and x > s2x:
                    flag = -1
            else:
                if x < s1x and x < s2x:
                    flag = -1

        if flag != 0:
            # the intersection point is either aft of both bows or forward of both sterns
            # determine which boat is to windward at the intersection point
            # add a small amount to x closer to the boats locations and check which boat is further to windward at this x location
            if state[idx_b1cwa] > 0:
                x += flag
            else:
                x -= flag
            m = 1. / tan(state[idx_b1cwa])
            b1y = state[idx_b1y] + m * (x - state[idx_b1x])
            m = 1. / tan(state[idx_b2cwa])
            b2y = state[idx_b2y] + m * (x - state[idx_b2x])
            if b2y > b1y:
                # boat 2 is the windward boat
                return 1
            else:
                # boat 1 is the windward boat
                return -1

    return 0

################################################################
cdef float line_intersection_x(float x1, float y1, float cwa1, float x2, float y2, float cwa2):
    cdef float m1, m2, c1, c2, x
    m1 = 1. / tan(cwa1)
    m2 = 1. / tan(cwa2)
    c1 = y1 - m1 * x1
    c2 = y2 - m2 * x2
    x = (c1 - c2) / (m2 - m1)
    return x

################################################################
cdef float line_segment_intersect(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4):
    cdef float denom, n_a, n_b, mu_a, mu_b, intersect
    intersect = -1
    denom = (y4-y3) * (x2-x1) - (x4-x3) * (y2-y1)
    if denom == 0:
        denom = 1e-6
    n_a = (x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)
    n_b = (x2-x1) * (y1-y3) - (y2-y1) * (x1-x3)
    mu_a = n_a / denom
    mu_b = n_b / denom
    if mu_a >= 0 and mu_a <= 1 and mu_b >= 0 and mu_b <= 1:
        intersect = mu_b
    return intersect

################################################################
cdef bint overlap(float[::1] state, float [:,::1] boundary):
    cdef float x1, x2, x3, x4, y1, y2, y3, y4, mu_a, mu_b, denom, n_a, n_b, dist
    cdef int i, i2, j, j2

    dist = sqrt((state[idx_b2x] - state[idx_b1x]) ** 2 + (state[idx_b2y] - state[idx_b1y]) ** 2)
    if dist < 30:
        # step through lines that make up boat 1 boundary
        for i in range(4):
            x1 = state[idx_b1x] + boundary[i,0] * cos(state[idx_b1cwa]) + boundary[i,1] * sin(state[idx_b1cwa])
            y1 = state[idx_b1y] + boundary[i,1] * cos(state[idx_b1cwa]) - boundary[i,0] * sin(state[idx_b1cwa])
            i2 = i + 1
            if i2 == 4:
                i2 = 0
            x2 = state[idx_b1x] + boundary[i2,0] * cos(state[idx_b1cwa]) + boundary[i2,1] * sin(state[idx_b1cwa])
            y2 = state[idx_b1y] + boundary[i2,1] * cos(state[idx_b1cwa]) - boundary[i2,0] * sin(state[idx_b1cwa])
            # step through lines that make up boat 2 boundary
            for j in range(4):
                x3 = state[idx_b2x] + boundary[j,0] * cos(state[idx_b2cwa]) + boundary[j,1] * sin(state[idx_b2cwa])
                y3 = state[idx_b2y] + boundary[j,1] * cos(state[idx_b2cwa]) - boundary[j,0] * sin(state[idx_b2cwa])
                j2 = j + 1
                if j2 == 4:
                    j2 = 0
                x4 = state[idx_b2x] + boundary[j2,0] * cos(state[idx_b2cwa]) + boundary[j2,1] * sin(state[idx_b2cwa])
                y4 = state[idx_b2y] + boundary[j2,1] * cos(state[idx_b2cwa]) - boundary[j2,0] * sin(state[idx_b2cwa])

                # check if the lines intersect
                if (line_segment_intersect(x1, y1, x2, y2, x3, y3, x4, y4) >= 0):
                    return 1

    return 0

################################################################
cdef over_line(float[::1] state, int idx_x, int idx_y, int cwa, float [:,::1] b, float[::1] over):
    cdef float y, m, vertex_x, vertex_y
    cdef int i
    over[0] = -10000
    over[1] = 0
    m = (state[idx_stb_y] - state[idx_prt_y]) / (state[idx_stb_x] - state[idx_prt_x])

    for i in range(4):
        vertex_x = state[idx_x] + b[i,0] * cos(state[cwa]) + b[i,1] * sin(state[cwa])
        vertex_y = state[idx_y] + b[i,1] * cos(state[cwa]) - b[i,0] * sin(state[cwa])
        y = m * (vertex_x - state[idx_prt_x]) + state[idx_prt_y]
        if vertex_y - y > over[0]:
            over[0] = vertex_y - y
            over[1] = i
    return 
    
################################################################
cdef float final_dmg(float[::1] state, int idx_x, int idx_y, int idx_v, int idx_cwa, int idx_tr, int idx_started):
    # step forward 5 seconds, estimating final distance to windward
    # assuming the boat turns to upwind vmg angle
    cdef int i, n
    cdef float acc, req_cwa = vmg_cwa, turn_angle, x, y, v, cwa, tr
    x = state[idx_x]
    y = state[idx_y]
    v = state[idx_v]
    cwa = state[idx_cwa]
    tr = state[idx_tr]

    # get the vmg angle to turn to
    if cwa < 0:
        req_cwa *= -1

    # step forward 5 seconds
    for i in range(5):
        
        # update cwa
        turn_angle = req_cwa - cwa
        if fabs(turn_angle) > 1:
            turn_angle /= fabs(turn_angle)
        tr = calc_turn_rate(v, cwa, tr, turn_angle, 1.0)
        cwa += tr

        # acceleration
        acc = calc_acc(state[idx_tws], cwa, v, tr, 0)

        # update velocity
        v += acc

        # update position
        y += v * cos(cwa)

    # if the boat hasn't started, take away the time it will take to get to the line
    if state[idx_started] < 0.5:
        if state[idx_y] > 0:
            y -= 2 * state[idx_y]

        if fabs(x) > state[idx_stb_x]:
            y -= (fabs(x) - state[idx_stb_x]) * 0.6

    return y

################################################################
def game_reward(float[::1] state, float[::1] prev_state, int row, float dt, float prestart_duration):
    cdef float  r = 0, b2b1_x, b2b1_y, b2b1_dir
    cdef float b1_over, b2_over, d
    cdef int vertex, prev_vertex
    info = {}

    #v_loc_np = np.array([0, 0], dtype=np.float32)
    #prev_v_loc_np = np.array([0, 0], dtype=np.float32)
    #cdef float [::1] v_loc = v_loc_np
    #cdef float [::1] prev_v_loc = prev_v_loc_np    

    cdef array.array v_loc_ = array.array('f', [0, 0])
    cdef array.array prev_v_loc_ = array.array('f', [0, 0])
    cdef array.array over_ = array.array('f', [0, 0])
    cdef array.array prev_over_ = array.array('f', [0, 0])    
    cdef float[::1] v_loc = v_loc_
    cdef float[::1] prev_v_loc = prev_v_loc_
    cdef float[::1] over = over_
    cdef float[::1] prev_over = prev_over_  

    ################################################################
    # check for infringement
    if overlap(state, virtual_boundary):
        if not overlap(prev_state, virtual_boundary):
            pnlt_freq = 1
            if row < 0:
                r -= game_penalty
                info["b1_penalty"] = {"type": "collision penalty"}
            else:
                r += game_penalty
                info["b2_penalty"] = {"type": "collision penalty"}

        # check for collisions
        if overlap(state, physical_boundary):
            # if collided, kill episode directly for faster iteration
            return float(r), info, True

    ################################################################
    # check for staying in the start box
    if state[idx_b1y] < -box_depth and prev_state[idx_b1y] > -box_depth:
        r -= game_penalty
        info["b1_penalty"] = {"type": "out of bounds"}
    if fabs(state[idx_b1x]) > box_width and fabs(prev_state[idx_b1x]) < box_width:
        r -= game_penalty
        info["b1_penalty"] = {"type": "out of bounds"}

    if state[idx_b2y] < -box_depth and prev_state[idx_b2y] > -box_depth:
        r += game_penalty
        info["b2_penalty"] = {"type": "out of bounds"}
    if fabs(state[idx_b2x]) > box_width and fabs(prev_state[idx_b2x]) < box_width:
        r += game_penalty
        info["b2_penalty"] = {"type": "out of bounds"}

    ################################################################
    # check for entering the start box correctly 
    # boat 1
    if state[idx_b1ent] < 0.5:
        over_line(state, idx_b1x, idx_b1y, idx_b1cwa, virtual_boundary, over)
        over_line(prev_state, idx_b1x, idx_b1y, idx_b1cwa, virtual_boundary, prev_over)
        if over[0] < 0 and prev_over[0] > 0 and fabs(state[idx_b1x]) < state[idx_stb_x]:
            # entered correctly
            state[idx_b1ent] = 1

        if state[idx_b1x] < state[idx_stb_x] and prev_state[idx_b1x] > state[idx_stb_x]:
            # crossed entry mark
            if state[idx_b1y] < state[idx_stb_y]:
                # didn't enter above the entry mark, penalise it
                r -= start_penalty + (state[idx_stb_y] - state[idx_b1y]) / 1000.0
                info["b1_penalty"] = {"type": "didn't enter correctly"}
        if state[idx_b1x] > state[idx_prt_x] and prev_state[idx_b1x] < state[idx_prt_x]:
            # crossed entry mark
            if state[idx_b1y] < state[idx_prt_y]:
                # didn't enter above the entry mark, penalise it
                r -= start_penalty + (state[idx_prt_y] - state[idx_b1y]) / 1000.0
                info["b1_penalty"] = {"type": "didn't enter correctly"}
        if state[idx_t] > 30:
            # didn't enter within 30s, penalise it
            if state[idx_b1x] > 0:
                d = sqrt((state[idx_b1x] - state[idx_stb_x]) ** 2 + (state[idx_b1y] - state[idx_stb_y]) ** 2)
                r -= (start_penalty + d / 1000.)
                info["b1_penalty"] = {"type": "didn't enter"}
            else:
                d = sqrt((state[idx_b1x] - state[idx_prt_x]) ** 2 + (state[idx_b1y] - state[idx_prt_y]) ** 2)
                r -= (start_penalty + d / 1000.)
                info["b1_penalty"] = {"type": "didn't enter"}
            state[idx_b1ent] = 1

    # boat 2
    if state[idx_b2ent] < 0.5:
        over_line(state, idx_b2x, idx_b2y, idx_b2cwa, virtual_boundary, over)
        over_line(prev_state, idx_b2x, idx_b2y, idx_b2cwa, virtual_boundary, prev_over)
        if over[0] < 0 and prev_over[0] > 0 and fabs(state[idx_b2x]) < state[idx_stb_x]:
            # entered correctly
            state[idx_b2ent] = 1

        if state[idx_b2x] < state[idx_stb_x] and prev_state[idx_b2x] > state[idx_stb_x]:
            # crossed entry mark
            if state[idx_b2y] < state[idx_stb_y]:
                # didn't enter above the entry mark, penalise it
                r += start_penalty + (state[idx_stb_y] - state[idx_b2y]) / 1000.0
                info["b2_penalty"] = {"type": "didn't enter correctly"}
        if state[idx_b2x] > state[idx_prt_x] and prev_state[idx_b2x] < state[idx_prt_x]:
            # crossed entry mark
            if state[idx_b2y] < state[idx_prt_y]:
                # didn't enter above the entry mark, penalise it
                r += start_penalty + (state[idx_prt_y] - state[idx_b2y]) / 1000.0
                info["b2_penalty"] = {"type": "didn't enter correctly"}
        if state[idx_t] > 30:
            # didn't enter within 30s, penalise it
            if state[idx_b2x] > 0:
                d = sqrt((state[idx_b2x] - state[idx_stb_x]) ** 2 + (state[idx_b2y] - state[idx_stb_y]) ** 2)
                r += (start_penalty + d / 1000.)
                info["b2_penalty"] = {"type": "didn't enter"}
            else:
                d = sqrt((state[idx_b2x] - state[idx_prt_x]) ** 2 + (state[idx_b2y] - state[idx_prt_y]) ** 2)
                r += (start_penalty + d / 1000.)
                info["b2_penalty"] = {"type": "didn't enter"}
            state[idx_b2ent] = 1

    ################################################################
    # if the start has happened
    if state[idx_t] > prestart_duration:

        # check for the boats starting
        # boat 1
        if state[idx_b1start] < 0.5:
            over_line(state, idx_b1x, idx_b1y, idx_b1cwa, virtual_boundary, over)
            if over[0] > 0:
                over_line(prev_state, idx_b1x, idx_b1y, idx_b1cwa, virtual_boundary, prev_over)
                if prev_over[0] <= 0:
                    get_vertex_position(state[idx_b1x], state[idx_b1y], state[idx_b1cwa], over[1], virtual_boundary, v_loc)
                    get_vertex_position(prev_state[idx_b1x], prev_state[idx_b1y], prev_state[idx_b1cwa], over[1], virtual_boundary, prev_v_loc)
                    if line_segment_intersect(v_loc[0], v_loc[1], prev_v_loc[0], prev_v_loc[1], state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y]) >= 0:
                        # started correctly
                        state[idx_b1start] = 1

        # boat 2
        if state[idx_b2start] < 0.5:
            over_line(state, idx_b2x, idx_b2y, idx_b2cwa, virtual_boundary, over)
            if over[0] > 0:
                over_line(prev_state, idx_b2x, idx_b2y, idx_b2cwa, virtual_boundary, prev_over)
                if prev_over[0] <= 0:
                    get_vertex_position(state[idx_b2x], state[idx_b2y], state[idx_b2cwa], over[1], virtual_boundary, v_loc)
                    get_vertex_position(prev_state[idx_b2x], prev_state[idx_b2y], prev_state[idx_b2cwa], over[1], virtual_boundary, prev_v_loc) 
                    if line_segment_intersect(v_loc[0], v_loc[1], prev_v_loc[0], prev_v_loc[1], state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y]) >= 0:
                        # started correctly
                        state[idx_b2start] = 1

        # check for either boat getting to 50m dmg after the start
        if (state[idx_b1y] >= dmg_after_start and state[idx_b1start] > 0.5) or (state[idx_b2y] >= dmg_after_start and state[idx_b2start] > 0.5):
            r += final_reward_scale * final_dmg(state, idx_b1x, idx_b1y, idx_b1v, idx_b1cwa, idx_b1tr, idx_b1start) / 1000.0
            r -= final_reward_scale * final_dmg(state, idx_b2x, idx_b2y, idx_b2v, idx_b2cwa, idx_b2tr, idx_b2start) / 1000.0
            return float(r), info, True

        # check for the episode ending because of time running out
        if state[idx_t] >= prestart_duration + t_after_start:
            r += final_reward_scale * final_dmg(state, idx_b1x, idx_b1y, idx_b1v, idx_b1cwa, idx_b1tr, idx_b1start) / 1000.0
            r -= final_reward_scale * final_dmg(state, idx_b2x, idx_b2y, idx_b2v, idx_b2cwa, idx_b2tr, idx_b2start) / 1000.0
            return float(r), info, True

    return float(r), info, False

################################################################
def step(float[::1] state, float[::1] ini_state, float[::1] action, int row, float dt, float prestart_duration, float ave_tws):
    cdef float acc

    ##################################################################
    # tws
    state[idx_tws] += rand_float() * dt * 2 + 0.05 * (ave_tws - state[idx_tws]) * dt
    if state[idx_tws] < 3.5:
        state[idx_tws] = 3.5
    if state[idx_tws] > 11.5:
        state[idx_tws] = 11.5

    ##################################################################
    # boat 1
    # update cwa
    state[idx_b1tr] = calc_turn_rate(state[idx_b1v], state[idx_b1cwa], state[idx_b1tr], action[0], dt)
    state[idx_b1cwa] += state[idx_b1tr] * dt
    state[idx_b1cwa] = limit_pi(state[idx_b1cwa])

    # longitudinal acceleration
    acc = calc_acc(state[idx_tws], state[idx_b1cwa], state[idx_b1v], state[idx_b1tr], action[1])

    # update velocity
    state[idx_b1v] += acc * dt

    # update position
    state[idx_b1x] += state[idx_b1v] * sin(state[idx_b1cwa]) * dt
    state[idx_b1y] += state[idx_b1v] * cos(state[idx_b1cwa]) * dt

    ##################################################################
    # boat 2
    # update cwa
    state[idx_b2tr] = calc_turn_rate(state[idx_b2v], state[idx_b2cwa], state[idx_b2tr], action[2], dt)
    state[idx_b2cwa] += state[idx_b2tr] * dt  
    state[idx_b2cwa] = limit_pi(state[idx_b2cwa])

    # longitudinal acceleration
    acc = calc_acc(state[idx_tws], state[idx_b2cwa], state[idx_b2v], state[idx_b2tr], action[3])    

    # update velocity
    state[idx_b2v] += acc * dt

    # update position
    state[idx_b2x] += state[idx_b2v] * sin(state[idx_b2cwa]) * dt
    state[idx_b2y] += state[idx_b2v] * cos(state[idx_b2cwa]) * dt

    ##################################################################
    # time
    state[idx_t] += dt

    ##################################################################
    # calculate the reward for this state for boat 1
    r, info, done = game_reward(state, ini_state, row, dt, prestart_duration)

    ##################################################################
    # calculate the current right of way
    row = calc_row(state)

    return r, done, info, int(row)

################################################################
def normalise(float[::1] state, float  prestart_duration):
    # boat 1
    state[idx_b1x] /= 1000
    state[idx_b1y] /= 1000
    state[idx_b1v] *= 0.06
    state[idx_b1cwa] /= pi

    # boat 2
    state[idx_b2x] /= 1000
    state[idx_b2y] /= 1000
    state[idx_b2v] *= 0.06
    state[idx_b2cwa] /= pi

    # time
    state[idx_t] -= prestart_duration
    state[idx_t] /= 60
    state[idx_t] += 1

    # marks
    state[idx_prt_x] /= 1000
    state[idx_prt_y] /= 1000
    state[idx_stb_x] /= 1000
    state[idx_stb_y] /= 1000

    # tws
    state[idx_tws] -= 7.5
    state[idx_tws] /= 4

    return

################################################################
def game_reset(float[::1] state, float length, float skew, b1_enter_stb):
    cdef int entry_side = 1, row
    cdef float d, ang
    if not b1_enter_stb:
        entry_side = -1

    ##################################################################
    # marks
    state[idx_prt_x] = -length * cos(skew) / 2
    state[idx_prt_y] = -length * sin(skew) / 2
    state[idx_stb_x] = length * cos(skew) / 2
    state[idx_stb_y] = length * sin(skew) / 2

    ##################################################################
    # tws
    state[idx_tws] = rand_float_0_1() * 8 + 3.5

    ##################################################################
    # boat 1
    
    # cwa
    state[idx_b1cwa] = -deg_to_rad(100.0) * entry_side + rand_float() * deg_to_rad(30.0)

    # turn rate
    state[idx_b1tr] = 0

    # v
    state[idx_b1v] = calc_polar_v(state[idx_tws], state[idx_b1cwa])

    # location
    if entry_side == 1:
        d = state[idx_b1v] * (10 + rand_float_0_1() * 3)
        ang = state[idx_b1cwa] + pi + rand_float() * deg_to_rad(30.0)
        state[idx_b1x] = state[idx_stb_x] + d * sin(ang)
        state[idx_b1y] = state[idx_stb_y] + d * cos(ang) + 20
    else:
        d = state[idx_b1v] * rand_float_0_1() * 3
        ang = state[idx_b1cwa] + pi + rand_float() * deg_to_rad(30.0)
        state[idx_b1x] = state[idx_prt_x] + d * sin(ang)
        state[idx_b1y] = state[idx_prt_y] + d * cos(ang) + 20

    # entered
    state[idx_b1ent] = 0

    # started
    state[idx_b1start] = 0

    ##################################################################
    # boat 2
    
    # cwa
    state[idx_b2cwa] = deg_to_rad(100.0) * entry_side + rand_float() * deg_to_rad(30.0)

    # turn rate
    state[idx_b2tr] = 0

    # v
    state[idx_b2v] = calc_polar_v(state[idx_tws], state[idx_b2cwa])

    # location
    if entry_side == -1:
        d = state[idx_b2v] * (10 + rand_float_0_1() * 3)
        ang = state[idx_b2cwa] + pi + rand_float() * deg_to_rad(30.0)
        state[idx_b2x] = state[idx_stb_x] + d * sin(ang)
        state[idx_b2y] = state[idx_stb_y] + d * cos(ang) + 20
    else:
        d = state[idx_b2v] * rand_float_0_1() * 3
        ang = state[idx_b2cwa] + pi + rand_float() * deg_to_rad(30.0)
        state[idx_b2x] = state[idx_prt_x] + d * sin(ang)
        state[idx_b2y] = state[idx_prt_y] + d * cos(ang) + 20

    # entered
    state[idx_b2ent] = 0

    # started
    state[idx_b2start] = 0

    ##################################################################
    # time
    state[idx_t] = 0

    ##################################################################
    # right of way
    row = calc_row(state)
    state[idx_row] = row
    state[idx_row_1] = row
    state[idx_row_2] = row

    return 

##################################################################