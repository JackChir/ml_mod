#ifndef __MHA_T_HPP__
#define __MHA_T_HPP__
#include <vector>

#include "mat.hpp"
#include "bp.hpp"
#include "activate_function.hpp"
#include "base_net.hpp"

namespace mha{

// 生成一个头部，列数量对应的是数据的数量，token_len对应的是token长度
template<int token_len, int data_num, typename val_t = double>
struct header_gen
{
    using type = mat<token_len, data_num, val_t>;
    bp<val_t, data_num, nadam, no_activate, XavierGaussian, token_len, token_len> Wq;  // Query权重矩阵
    bp<val_t, data_num, nadam, no_activate, XavierGaussian, token_len, token_len> Wk;  // Key权重矩阵
    bp<val_t, data_num, nadam, no_activate, XavierGaussian, token_len, token_len> Wv;  // Value权重矩阵
    softmax<mat<data_num, data_num, val_t>> softmax_func;  // Softmax激活函数
    mat<data_num, data_num, val_t> softmax_output;  // 上次输出的注意力分数矩阵
    mat<token_len, data_num, val_t> Q;  // Query矩阵
    mat<token_len, data_num, val_t> K;  // Key矩阵
    mat<token_len, data_num, val_t> V;


    mat<token_len, data_num, val_t> forward(const mat<token_len, data_num, val_t>& input, const bool domask = false)
    {
        // 计算Query、Key和Value
        Q = Wq.forward(input);         // Q类型mat<token_len, data_num, val_t>
        K = Wk.forward(input);         // K类型mat<token_len, data_num, val_t>
        V = Wv.forward(input);         // V类型mat<token_len, data_num, val_t>

        auto sqrt_QtK = Q.t().dot(K) / std::sqrt(static_cast<double>(token_len));  // 计算Q和K的点积，得到注意力分数矩阵
        if (domask)
        {
            // 如果需要掩码处理，可以在这里添加掩码逻辑
            // 例如，将某些位置的分数设置为负无穷大，以避免它们在softmax中被考虑
            // 这通常用于处理自注意力中的未来信息泄露问题
            for (int i = 0; i < data_num; ++i)
            {
                for (int j = i + 1; j < data_num; ++j)
                {
                    sqrt_QtK.get(i, j) = -std::numeric_limits<val_t>::infinity();  // 设置未来位置为负无穷大
                }
            }
        }
        // 计算注意力权重
        softmax_output = softmax_func.forward(sqrt_QtK);  // 缩放
        // scores类型 mat<token_len, token_len, val_t>

        return V.dot(softmax_output.t());  // 返回经过注意力机制处理后的输出
    }

    mat<token_len, data_num, val_t> backward(const mat<token_len, data_num, val_t>& delta)
    {
        // 求出误差对V的梯度
        auto dV = delta.dot(softmax_output);  // deltaV类型 mat<token_len, data_num, val_t>
        auto deltaSoftmax = delta.t().dot(V);         // deltaSoftmax类型 mat<token_len, token_len, val_t>
        auto deltaQK = deltaSoftmax*softmax_func.backward();
        /**
          对于$$C=A\cdot B$$，误差反向传播对A和B的偏导数为：
          $$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T$$
          $$\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$$
          由于计算的是Q \cdot K^T，所以对Q的梯度为：
          $$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial C} \cdot K$$
          对K的梯度为：
          $$\frac{\partial L}{\partial K} = (\frac{\partial L}{\partial C})^T \cdot Q$$
         */
        auto dQ = K.dot(deltaQK.t()) / std::sqrt(static_cast<double>(token_len));  // Q的梯度
        auto dK = Q.dot(deltaQK) / std::sqrt(static_cast<double>(token_len));  // K的梯度

        auto deltaV = Wv.backward(dV);  // 更新V的权重，并反馈V的误差
        auto deltaQ = Wq.backward(dQ);  // 更新Q的权重，并反馈Q的误差
        auto deltaK = Wk.backward(dK);  // 更新K的权重，并反馈K的误差
        // 返回误差
        return deltaV + deltaQ + deltaK;  // 返回误差
    }

    void update_inert()
    {
        Wq.update_inert();  // 更新Q的权重
        Wk.update_inert();  // 更新K的权重
        Wv.update_inert();  // 更新V的权重
    }
};

template<int token_len, int data_num, int header_num, typename val_t = double>
struct mha_t
{
    using input_type = mat<token_len, data_num, val_t>;  // 输入类型
    using ret_type = mat<token_len, data_num, val_t>;  // 返回类型
    std::vector<header_gen<token_len, data_num, val_t>> headers;  // 多头注意力机制的多个头部
    mat<header_num, 1, input_type> header_outputs;  // 每个头部的输出
    bp<input_type, 1, nadam, ReLu, XavierGaussian, header_num, 1> WReLu;
    bool domask;  // 是否使用掩码

    mha_t():domask(false)
    {
        headers.resize(header_num);
    }

    mat<token_len, data_num, val_t> forward(const mat<token_len, data_num, val_t>& input)
    {
        // 使用输入对每个多头注意力头进行前向传播
        for (int i = 0; i < header_num; ++i)
        {
            header_outputs.get(i, 0) = headers[i].forward(input, domask);  // 获取每个头部的输出
        }
        return WReLu.forward(header_outputs)[0];  // 将所有头部的输出通过ReLU和归一化层
    }

    mat<token_len, data_num, val_t> backward(const mat<token_len, data_num, val_t>& delta)
    {
        mat<1, 1, input_type> delta_out;
        delta_out.get(0, 0) = delta;  // 将delta转换为适合WReLu的格式
        auto delta_WReLu = WReLu.backward(delta_out);  // 反向传播到ReLU层
        // 返回每个头部的输出误差的和
        mat<token_len, data_num, val_t> delta_sum;
        for (int i = 0; i < header_num; ++i)
        {
            delta_sum = delta_sum + headers[i].backward(delta_WReLu.get(i, 0));  // 累加每个头部的输出误差
        }
        return delta_sum;  // 返回总的误差
    }
    void update_inert()
    {
        for (int i = 0; i < header_num; ++i)
        {
            headers[i].update_inert();  // 更新每个头部的权重
        }
        WReLu.update_inert();  // 更新ReLU层的权重
    }
};

// 交叉注意力，用于结合编码器和解码器的输出
template<int token_len, int encoder_data_num, int decoder_data_num, typename val_t = double>
struct cross_header_gen
{
    using encoder_input_type = mat<token_len, encoder_data_num, val_t>;
    using decoder_input_type = mat<token_len, decoder_data_num, val_t>;
    using ret_type = mat<token_len, decoder_data_num, val_t>;  // 返回类型
    bp<val_t, decoder_data_num, nadam, no_activate, XavierGaussian, token_len, token_len> Wq;  // Query权重矩阵
    bp<val_t, encoder_data_num, nadam, no_activate, XavierGaussian, token_len, token_len> Wk;  // Key权重矩阵
    bp<val_t, encoder_data_num, nadam, no_activate, XavierGaussian, token_len, token_len> Wv;  // Value权重矩阵
    softmax<mat<encoder_data_num, decoder_data_num, val_t>> softmax_func;  // Softmax激活函数
    mat<encoder_data_num, decoder_data_num, val_t> softmax_output;  // 上次输出的注意力分数矩阵
    mat<token_len, decoder_data_num, val_t> Q;  // Query矩阵
    mat<token_len, encoder_data_num, val_t> K;  // Key矩阵
    mat<token_len, encoder_data_num, val_t> V;


    ret_type forward(const encoder_input_type& encoder_input, const decoder_input_type& decoder_input)
    {
        // 计算Query、Key和Value
        Q = Wq.forward(decoder_input);         // Q类型mat<token_len, data_num, val_t>
        K = Wk.forward(encoder_input);         // K类型mat<token_len, data_num, val_t>
        V = Wv.forward(encoder_input);         // V类型mat<token_len, data_num, val_t>
        auto sqrt_QtK = Q.t().dot(K) / std::sqrt(static_cast<double>(token_len));  // 计算Q和K的点积，得到注意力分数矩阵
        softmax_output = softmax_func.forward(sqrt_QtK);  // 缩放
        return V.dot(softmax_output.t());  // 返回经过注意力机制处理后的输出
    }

    void backward(const ret_type& delta, encoder_input_type& encoder_delta, decoder_input_type& decoder_delta)
    {
        // 求出误差对V的梯度
        auto dV = delta.dot(softmax_output);  // deltaV类型 mat<token_len, data_num, val_t>
        auto deltaSoftmax = delta.t().dot(V);         // deltaSoftmax类型 mat<token_len, token_len, val_t>
        auto deltaQK = deltaSoftmax*softmax_func.backward();
        /**
          对于$$C=A\cdot B$$，误差反向传播对A和B的偏导数为：
          $$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T$$
          $$\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$$
          由于计算的是Q \cdot K^T，所以对Q的梯度为：
          $$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial C} \cdot K$$
          对K的梯度为：
          $$\frac{\partial L}{\partial K} = (\frac{\partial L}{\partial C})^T \cdot Q$$
         */
        auto dQ = K.dot(deltaQK.t()) / std::sqrt(static_cast<double>(token_len));  // Q的梯度
        auto dK = Q.dot(deltaQK) / std::sqrt(static_cast<double>(token_len));  // K的梯度

        auto deltaV = Wv.backward(dV);  // 更新V的权重，并反馈V的误差
        decoder_delta = Wq.backward(dQ);  // 更新Q的权重，并反馈Q的误差
        auto deltaK = Wk.backward(dK);  // 更新K的权重，并反馈K的误差
        encoder_delta = deltaV + deltaK;  // 返回误差
    }

    void update_inert()
    {
        Wq.update_inert();  // 更新Q的权重
        Wk.update_inert();  // 更新K的权重
        Wv.update_inert();  // 更新V的权重
    }
};

template<int token_len, int encoder_data_num, int decoder_data_num, int header_num, typename val_t = double>
struct cross_mha_t
{
    using encoder_input_type = mat<token_len, encoder_data_num, val_t>;  // 输入类型
    using decoder_input_type = mat<token_len, decoder_data_num, val_t>;  // 输入类型
    using ret_type = mat<token_len, decoder_data_num, val_t>;  // 返回类型
    using head_gen_t = cross_header_gen<token_len, encoder_data_num, decoder_data_num, val_t>;
    std::vector<head_gen_t> headers;  // 多头注意力机制的多个头部
    mat<header_num, 1, decoder_input_type> header_outputs;  // 每个头部的输出，会被串成一列送入BP
    bp<decoder_input_type, 1, nadam, ReLu, XavierGaussian, header_num, 1> WReLu;

    cross_mha_t()
    {
        headers.resize(header_num);
    }

    ret_type forward(const encoder_input_type& encoder_input, const decoder_input_type& decoder_input)
    {
        // 使用输入对每个多头注意力头进行前向传播
        for (int i = 0; i < header_num; ++i)
        {
            header_outputs.get(i, 0) = headers[i].forward(encoder_input, decoder_input);  // 获取每个头部的输出
        }
        return WReLu.forward(header_outputs)[0];  // 将所有头部的输出通过ReLU和归一化层
    }

    void backward(const ret_type& delta, encoder_input_type& encoder_delta, decoder_input_type& decoder_delta)
    {
        mat<1, 1, ret_type> delta_out;
        delta_out.get(0, 0) = delta;  // 将delta转换为适合WReLu的格式
        auto delta_WReLu = WReLu.backward(delta_out);  // 反向传播到ReLU层
        // 返回每个头部的输出误差的和
        encoder_input_type encoder_delta_cur;
        decoder_input_type decoder_delta_cur;
        encoder_delta = 0.;
        decoder_delta = 0.;
        for (int i = 0; i < header_num; ++i)
        {
            headers[i].backward(delta_WReLu.get(i, 0), encoder_delta_cur, decoder_delta_cur);  // 累加每个头部的输出误差
            encoder_delta += encoder_delta_cur;
            decoder_delta += decoder_delta_cur;
        }
    }
    void update_inert()
    {
        for (int i = 0; i < header_num; ++i)
        {
            headers[i].update_inert();  // 更新每个头部的权重
        }
        WReLu.update_inert();  // 更新ReLU层的权重
    }
};

} // namespace mha

template<int token_len, int data_num, typename val_t = double>
void write_file(const mha::header_gen<token_len, data_num, val_t>& header, ht_memory& mry)
{
    write_file(header.Wq, mry);
    write_file(header.Wk, mry);
    write_file(header.Wv, mry);
    write_file(header.softmax_func, mry);
}

template<int token_len, int data_num, typename val_t = double>
void read_file(ht_memory& mry, mha::header_gen<token_len, data_num, val_t>& header)
{
    read_file(mry, header.Wq);
    read_file(mry, header.Wk);
    read_file(mry, header.Wv);
    read_file(mry, header.softmax_func);
}

template<int token_len, int data_num, int header_num, typename val_t = double>
void write_file(const mha::mha_t<token_len, data_num, header_num, val_t>& mha, ht_memory& mry)
{
    for (int i = 0; i < header_num; ++i)
    {
        write_file(mha.headers[i], mry);  // 将每个头部写入文件
    }
    write_file(mha.WReLu, mry);  // 将ReLU层写入文件
}
template<int token_len, int data_num, int header_num, typename val_t = double>
void read_file(ht_memory& mry, mha::mha_t<token_len, data_num, header_num, val_t>& mha)
{
    for (int i = 0; i < header_num; ++i)
    {
        read_file(mry, mha.headers[i]);  // 从文件中读取每个头部
    }
    read_file(mry, mha.WReLu);  // 从文件中读取ReLU层
}

#endif
