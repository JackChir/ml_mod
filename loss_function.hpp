#ifndef __LOSS_FUNCTION_HPP__
#define __LOSS_FUNCTION_HPP__

template<typename name>
struct loss_function
{
    template<typename output_t>
    static output_t cal(const output_t& output, const output_t& expected)
    {
        // 默认实现为0
        return 0.0;
    }
};

template<>
struct loss_function<class mse>
{
    // 计算均方误差相对于输入的偏导数
    template<typename output_t>
    static output_t cal(const output_t& output, const output_t& expected)
    {
        // 计算均方误差
        output_t diff = output - expected;
        // 返回偏导数
        constexpr double factor = 2.0 / (output_t::r * output_t::c);
        return diff * factor;
    }
};

// 交叉熵损失函数
template<>
struct loss_function<class cross_entropy>
{
    // 计算交叉熵损失相对于输入的偏导数
    template<typename output_t>
    static output_t cal(const output_t& output, const output_t& expected)
    {
        // 计算交叉熵损失
        output_t diff = output - expected;
        // 返回偏导数
        return diff / (output * (1.0 - output));
    }
};

#endif
