namespace {
    __device__ const float bloom_kernel[] = {
        0.000000046f, // [ 0]: -31
        0.000000108f, // [ 1]: -30
        0.000000249f, // [ 2]: -29
        0.000000559f, // [ 3]: -28
        0.000001227f, // [ 4]: -27
        0.000002629f, // [ 5]: -26
        0.000005499f, // [ 6]: -25
        0.000011221f, // [ 7]: -24
        0.000022331f, // [ 8]: -23
        0.000043323f, // [ 9]: -22
        0.000081900f, // [10]: -21
        0.000150817f, // [11]: -20
        0.000270421f, // [12]: -19
        0.000471941f, // [13]: -18
        0.000801355f, // [14]: -17
        0.001323381f, // [15]: -16
        0.002124736f, // [16]: -15
        0.003315321f, // [17]: -14
        0.005025641f, // [18]: -13
        0.007398654f, // [19]: -12
        0.010574632f, // [20]: -11
        0.014668714f, // [21]: -10
        0.019742578f, // [22]:  -9
        0.025773914f, // [23]:  -8
        0.032629535f, // [24]:  -7
        0.040049335f, // [25]:  -6
        0.047648035f, // [26]:  -5
        0.054939346f, // [27]:  -4
        0.061382872f, // [28]:  -3
        0.066448634f, // [29]:  -2
        0.069688951f, // [30]:  -1
        0.070804194f, // [31]:   0
        0.069688951f, // [32]:   1
        0.066448634f, // [33]:   2
        0.061382872f, // [34]:   3
        0.054939346f, // [35]:   4
        0.047648035f, // [36]:   5
        0.040049335f, // [37]:   6
        0.032629535f, // [38]:   7
        0.025773914f, // [39]:   8
        0.019742578f, // [40]:   9
        0.014668714f, // [41]:  10
        0.010574632f, // [42]:  11
        0.007398654f, // [43]:  12
        0.005025641f, // [44]:  13
        0.003315321f, // [45]:  14
        0.002124736f, // [46]:  15
        0.001323381f, // [47]:  16
        0.000801355f, // [48]:  17
        0.000471941f, // [49]:  18
        0.000270421f, // [50]:  19
        0.000150817f, // [51]:  20
        0.000081900f, // [52]:  21
        0.000043323f, // [53]:  22
        0.000022331f, // [54]:  23
        0.000011221f, // [55]:  24
        0.000005499f, // [56]:  25
        0.000002629f, // [57]:  26
        0.000001227f, // [58]:  27
        0.000000559f, // [59]:  28
        0.000000249f, // [60]:  29
        0.000000108f, // [61]:  30
        0.000000046f, // [62]:  31
    };
}