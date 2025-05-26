#ifndef __MHA_T_HPP__
#define __MHA_T_HPP__
#include "mat.hpp"
#include "bp.hpp"
#include "activate_function.hpp"

namespace mha{

// 生成一个头部
template<int row_num, int col_num, typename val_t = double>
struct header_gen
{
    using type = mat<row_num, col_num, val_t>;
    bp<val_t, col_num, nadam, no_activate, XavierGaussian, row_num, row_num> Wq;  // Query权重矩阵
    bp<val_t, col_num, nadam, no_activate, XavierGaussian, row_num, row_num> Wk;  // Key权重矩阵
    bp<val_t, col_num, nadam, no_activate, XavierGaussian, row_num, row_num> Wv;  // Value权重矩阵
    softmax<mat<row_num, row_num, val_t>> softmax_func;  // Softmax激活函数
    mat<row_num, row_num, val_t> scores;  // 上次输出的注意力分数矩阵
    mat<row_num, col_num, val_t> Q;  // Query矩阵
    mat<row_num, col_num, val_t> K;  // Key矩阵
    mat<row_num, col_num, val_t> V;


    mat<row_num, col_num, val_t> forward(const mat<row_num, col_num, val_t>& input)
    {
        pre_in = input;  // 保存输入以便在反向传播中使用
        // 计算Query、Key和Value
        Q = Wq.forward(input);         // Q类型mat<row_num, col_num, val_t>
        K = Wk.forward(input);         // K类型mat<row_num, col_num, val_t>
        V = Wv.forward(input);         // V类型mat<row_num, col_num, val_t>

        // 计算注意力权重
        scores = Q.dot(K.t()) / std::sqrt(static_cast<double>(row_num));  // 缩放
        // scores类型 mat<row_num, row_num, val_t>

        return softmax_func.forward(scores).dot(V);  // 返回经过注意力机制处理后的输出
    }

    mat<row_num, col_num, val_t> backward(const mat<row_num, col_num, val_t>& delta)
    {
        // 求出误差对V的梯度
        auto dV = scores.t().dot(delta);
        auto dScores = delta.dot(V.t());
        auto deltaV = Wv.backward(dV);  // 更新V的权重，并反馈V的误差
        /**
         * 对于$$C=A\cdot B$$，误差反向传播对A和B的偏导数为：
         * $$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T$$
         * $$\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$$
         * 由于计算的是Q \cdot K^T，所以对Q的梯度为：
         * $$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial C} \cdot K$$
         * 对K的梯度为：
         * $$\frac{\partial L}{\partial K} = (\frac{\partial L}{\partial C})^T \cdot Q$$
         */
        auto dQ = dScores.dot(K) / std::sqrt(static_cast<double>(row_num));  // Q的梯度
        auto dK = dScores.t().dot(Q) / std::sqrt(static_cast<double>(row_num));  // K的梯度
        auto deltaQ = Wq.backward(dQ);  // 更新Q的权重，并反馈Q的误差
        auto deltaK = Wk.backward(dK);  // 更新K的权重，并反馈K的误差
        // 返回误差
        return deltaV + deltaQ + deltaK;  // 返回误差
    }
};

template<int row_num, int col_num, int header_num, typename val_t = double>
struct mha_t
{
    using input_type = mat<row_num, col_num, val_t>;  // 输入类型
    std::vector<header_gen<row_num, col_num, val_t>> headers;  // 多头注意力机制的多个头部
    mat<header_num, 1, input_type> header_outputs;  // 每个头部的输出
    bp<input_type, 1, nadam, ReLu, XavierGaussian, header_num, head_num> WReLu;
    bp<input_type, 1, nadam, act_norm, XavierGaussian, header_num, 1> WNorm;

    mha_t()
    {
        headers.resize(header_num);
    }

    mat<row_num, col_num, val_t> forward(const mat<row_num, col_num, val_t>& input)
    {
        // 使用输入对每个多头注意力头进行前向传播
        for (int i = 0; i < header_num; ++i)
        {
            header_outputs.get(i, 0) = headers[i].forward(input);  // 获取每个头部的输出
        }
        return WNorm.forward(WReLu.forward(header_outputs))[0];  // 将所有头部的输出通过ReLU和归一化层
    }

    mat<row_num, col_num, val_t> backward(const mat<row_num, col_num, val_t>& delta)
    {
        mat<1, 1, input_type> delta_out;
        delta_out.get(0, 0) = delta;  // 将delta转换为适合WReLu的格式
        auto delta_WNorm = WNorm.backward(delta_out);  // 反向传播到归一化层
        auto delta_WReLu = WReLu.backward(delta_WNorm);  // 反向传播到ReLU层
        // 对每个头部进行反向传播
        for (int i = 0; i < header_num; ++i)
        {
            headers[i].backward(delta_WReLu.get(i, 0));  // 将每个头部的输出误差传递给对应的头部
        }
        // 返回每个头部的输出误差的和
        mat<row_num, col_num, val_t> delta_sum;
        for (int i = 0; i < header_num; ++i)
        {
            delta_sum = delta_sum + headers[i].backward(delta_WReLu.get(i, 0));  // 累加每个头部的输出误差
        }
        return delta_sum;  // 返回总的误差
    }
};

}

#endif
