#ifndef __TRANSFORMER_T_HPP__
#define __TRANSFORMER_T_HPP__
#include "mha_t.hpp"
#include "base_net.hpp"

template<int token_len, int data_num, typename val_t = double>
struct transformer_unite_t
{
    using mha_type = residual_net<mha::mha_t<token_len, data_num, 1, val_t> >;  // 多头注意力机制类型
    using linear_type = residual_net<bp<val_t, data_num, nadam, no_activate, XavierGaussian, token_len, token_len> >;  // 线性层类型
    using input_type = mat<token_len, data_num, val_t>;  // 输入类型
    using ret_type = mat<token_len, data_num, val_t>;  // 返回类型

    mha_type mha;  // 多头注意力机制
    linear_type linear;  // 线性层
    transformer_unite_t()
    {
        // 初始化多头注意力机制和线性层
    }
    ret_type forward(const input_type& input)
    {
        auto mha_output = mha.forward(input);  // 前向传播到多头注意力机制
        return linear.forward(mha_output);  // 前向传播到线性层
    }

    input_type backward(const ret_type& delta)
    {
        auto delta_linear = linear.backward(delta);  // 反向传播到线性层
        return mha.backward(delta_linear);  // 反向传播到多头注意力机制
    }

};

template<int token_len, int encoder_data_num, int decoder_data_num, int header_num, typename val_t = double>
struct base_transformer_t
{
    using encoder_input_type = mat<token_len, encoder_data_num, val_t>;
    using decoder_input_type = mat<token_len, decoder_data_num, val_t>;
    using ret_type = mat<token_len, decoder_data_num, val_t>;  // 返回类型

    using encoder_unite_t = transformer_unite_t<token_len, encoder_data_num, val_t>;  // 编码器联合网络类型
    using decoder_unite_t = transformer_unite_t<token_len, decoder_data_num, val_t>;  // 解码器联合网络类型

    using cross_mha_t = residual_net<mha::cross_mha_t<token_len, encoder_data_num, decoder_data_num, header_num, val_t> >;  // 交叉注意力机制
    using cross_linear_t = residual_net<bp<val_t, decoder_data_num, nadam, no_activate, XavierGaussian, token_len, token_len> >;  // 交叉线性层

    using softmax_net_t = residual_net<bp<val_t, decoder_data_num, nadam, softmax_activate, XavierGaussian, token_len, token_len> >;  // Softmax网络

    std::vector<encoder_unite_t> encoder_units;  // 编码器联合网络的多个实例
    std::vector<decoder_unite_t> decoder_units;  // 解码器联合网络的多个实例

    cross_mha_t cross_mha;  // 交叉注意力机制
    cross_linear_t cross_linear;  // 交叉线性层

    softmax_net_t softmax_net;  // Softmax网络


    base_transformer_t(const int& stack_num)
    {
        encoder_units.resize(stack_num);  // 初始化编码器联合网络
        decoder_units.resize(stack_num);  // 初始化解码器联合网络
    }

    void switch_to_teacher_mode(const bool& teacher_mode_on = true)
    {
        for (auto& unit : decoder_units)
        {
            unit.mha.domask = teacher_mode_on;  // 设置多头注意力机制的掩码模式
        }
    }

    ret_type forward(const encoder_input_type& encoder_input, const decoder_input_type& decoder_input)
    {
        auto encoder_output = encoder_input;  // 编码器输入
        for (auto& unit : encoder_units)
        {
            encoder_output = unit.forward(encoder_output);  // 前向传播到每个编码器联合网络
        }

        auto decoder_output = decoder_input;  // 解码器输入
        for (auto& unit : decoder_units)
        {
            decoder_output = unit.forward(decoder_output);  // 前向传播到每个解码器联合网络
        }

        auto cross_output = cross_mha.forward(encoder_output, decoder_output);  // 交叉注意力机制
        auto cross_linear_output = cross_linear.forward(cross_output);  // 交叉线性层

        auto final_output = softmax_net.forward(cross_linear_output);  // Softmax网络
        return final_output;  // 返回最终输出
    }

    void backward(const ret_type& delta, encoder_input_type& encoder_delta, decoder_input_type& decoder_delta)
    {
        auto through_softmax = softmax_net.backward(delta);  // 反向传播到Softmax网络
        auto through_cross_linear = cross_linear.backward(through_softmax);  // 反向传播到交叉线性层
        cross_mha.backward(through_cross_linear, encoder_delta, decoder_delta);  // 反向传播到交叉注意力机制

        // 反向逐次流过编码器各层
        for (auto it = encoder_units.rbegin(); it != encoder_units.rend(); ++it)
        {
            encoder_delta = it->backward(encoder_delta);  // 反向传播到每个编码器联合网络
        }

        // 反向逐次流过解码器各层
        for (auto it = decoder_units.rbegin(); it != decoder_units.rend(); ++it)
        {
            decoder_delta = it->backward(decoder_delta);  // 反向传播到每个解码器联合网络
        }
    }

    void update_inert()
    {
        for (auto& unit : encoder_units)
        {
            unit.update_inert();  // 更新每个编码器联合网络的权重
        }
        for (auto& unit : decoder_units)
        {
            unit.update_inert();  // 更新每个解码器联合网络的权重
        }
        cross_mha.update_inert();  // 更新交叉注意力机制的权重
        cross_linear.update_inert();  // 更新交叉线性层的权重
        softmax_net.update_inert();  // 更新Softmax网络的权重
    }

};

// transformer_t是在base_transformer_t的基础上，增加了RoPE的支持，同时增加教师强制训练
template<int token_len, int encoder_data_num, int decoder_data_num, int header_num, typename val_t = double>
struct transformer_t : public base_transformer_t<token_len, encoder_data_num, decoder_data_num, header_num, val_t>
{
    using base_type = base_transformer_t<token_len, encoder_data_num, decoder_data_num, header_num, val_t>;
    using rope_precompute_type = RoPEPrecompute<token_len>;  // RoPE预计算类型
    rope_precompute_type rope_precompute;  // RoPE预计算实例

    transformer_t(const int& stack_num) : base_type(stack_num) {}

    auto forward_with_rope(const typename base_type::encoder_input_type& encoder_input,
                           const typename base_type::decoder_input_type& decoder_input,
                           const mat<encoder_data_num, 1, int>& mt_encoder_time, const mat<decoder_data_num, 1, int>& mt_decoder_time)
    {
        // 应用RoPE到编码器输入
        auto encoder_input_rope = encoder_input;
        rope_precompute.apply(encoder_input_rope, mt_encoder_time);

        // 应用RoPE到解码器输入
        auto decoder_input_rope = decoder_input;
        rope_precompute.apply(decoder_input_rope, mt_decoder_time);

        // 前向传播
        return this->forward(encoder_input_rope, decoder_input_rope);
    }

    // 教师强制学习，就是在decoder中输入真实的标签，同时开启teacher mode，然后使用forward_with_rope进行前向传播，并计算误差反向传播
    void train_with_teacher(const typename base_type::encoder_input_type& encoder_input,
                              const typename base_type::decoder_input_type& decoder_input,
                              const mat<encoder_data_num, 1, int>& mt_encoder_time, const mat<decoder_data_num, 1, int>& mt_decoder_time,
                              const typename base_type::ret_type& expected_output,
                              typename base_type::encoder_input_type& encoder_delta,
                              typename base_type::decoder_input_type& decoder_delta, const int& train_time = 100)
    {
        this->switch_to_teacher_mode(true);  // 开启教师模式
        for (int i = 0; i < train_time; ++i)
        {
            auto mode_output = this->forward_with_rope(encoder_input, decoder_input, mt_encoder_time, mt_decoder_time);  // 前向传播
            auto loss = loss_function<cross_entropy>::cal(mode_output, expected_output);  // 计算损失

            this->backward(loss, encoder_delta, decoder_delta);  // 反向传播
        }
        this->switch_to_teacher_mode(false);  // 关闭教师模式
    }
    // 将预测数据输入，在关闭教师模式的情况下进行前向传播。首先生成第一个token，然后将其作为下一个token的输入，直到生成指定长度的输出
    typename base_type::ret_type predict_with_rope(const typename base_type::encoder_input_type& encoder_input,
                                                   const mat<encoder_data_num, 1, int>& mt_encoder_time)
    {
        this->switch_to_teacher_mode(false);  // 确保关闭教师模式
        auto encoder_input_rope = encoder_input;
        rope_precompute.apply(encoder_input_rope, mt_encoder_time);  // 应用RoPE到编码器输入

        typename base_type::decoder_input_type decoder_input;  // 初始化解码器输入
        typename base_type::ret_type output;  // 初始化输出

        for (int i = 0; i < decoder_data_num; ++i)
        {
            // 向前传播
            auto mode_output = this->forward(encoder_input_rope, decoder_input);  // 前向传播
            output = mode_output;  // 更新输出
            decoder_input = mode_output;  // 将当前输出作为下一个解码器输入
            // 对新的输出执行RoPE
            rope_precompute.apply_to_col(decoder_input, i, mt_encoder_time.get(i, 0));  // 应用RoPE到解码器输入的当前列
            
        }
        return output;  // 返回最终输出
    }

    void update_inert()
    {
        base_type::update_inert();  // 更新基础网络的权重
        //rope_precompute.apply(base_type::encoder_units[0].mha.mha.Q, 0);  // 更新RoPE预计算
        //rope_precompute.apply(base_type::decoder_units[0].mha.mha.Q, 0);  // 更新RoPE预计算
    }
};

#endif
