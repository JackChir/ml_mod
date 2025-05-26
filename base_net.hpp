#ifndef __BASE_NET_HPP__
#define __BASE_NET_HPP__

template<int row_num, int col_num, typename val_t = double>
struct normalize_net
{
    mat<row_num, 1, val_t> norm_mean;  // 均值
    mat<row_num, 1, val_t> norm_div;   // 方差

    mat<row_num, col_num, val_t> forward(const mat<row_num, col_num, val_t>& input)
    {
        return normalize(input, norm_mean, norm_div);
    }

};

#endif
