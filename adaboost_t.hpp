#ifndef __ADABOOST_T_HPP__
#define __ADABOOST_T_HPP__

#include <vector>

template<typename val_t, typename weight_t = double>
struct type_with_weight
{
    val_t value;
    weight_t weight;

    type_with_weight() : value(0), weight(0) {}
    type_with_weight(const val_t& v, const weight_t& w) : value(v), weight(w) {}

    bool operator<(const type_with_weight& other) const
    {
        return value < other.value;
    }

    bool operator>(const type_with_weight& other) const
    {
        return value > other.value;
    }
};

// 需要实现的分类器接口
template<typename data_t>
class i_judger_t
{
public:
    // 对数据进行分类，得到一个结果
    virtual int judge(const data_t& data) const = 0;
};

template<typename data_t>
class adaboost_t
{
private:
    using judger_tp = i_judger_t<data_t>*; // 分类器类型
    using weight_judger_t = type_with_weight<judger_tp, double>; // 分类器和权重的组合
    using weight_data_t = type_with_weight<data_t, double>; // 数据和权重的组合
    std::vector<weight_judger_t> classifiers; // 分类器列表
    std::vector<weight_data_t> training_data; // 训练数据列表
    std::vector<int> expected_results; // 期望的分类结果
    int class_num; // 分类数量

    bool update_weights(const int& cidx)
    {
        // 使用第cidx个分类器对数据进行判断，计算出错误率
        if (cidx < 0 || cidx >= classifiers.size())
            return; // 如果索引不合法，直接返回
        judger_tp classifier = classifiers[cidx].value;
        double e = 0.0; // 错误率
        std::vector<bool> is_correct(training_data.size(), false); // 用于记录每个样本是否分类正确
        for (size_t i = 0; i < training_data.size(); ++i)
        {
            int result = classifier->judge(training_data[i].value);
            if (result != expected_results[i]) // 如果分类结果与期望结果不一致
            {
                e += training_data[i].weight; // 如果分类错误，累加权重
            }
            else // 如果分类正确
            {
                is_correct[i] = true; // 标记为正确
            }
        }
        if (e > (class_num - 1) / static_cast<double>(class_num))
        {
            return false; // 返回false表示分类器不合格
        }
        // 根据公式\alpha_t = 0.5 * log((1 - e) / (e + 1e-10))+log(K-1)计算分类器的权重，其中K为类别数
        double alpha_t = 0.5 * log((1 - e) / (e + 1e-10)) + log(class_num - 1);
        classifiers[cidx].weight = alpha_t; // 更新分类器的权重
        // 更新样本权重
        double sum_w = 0.0; // 样本权重归一化参数
        for (size_t i = 0; i < training_data.size(); ++i)
        {
            if (is_correct[i]) // 如果分类正确，则W_i^(t+1) = W_i^(t) * exp(-\alpha_t * \frac{K}{K-1})
            {
                training_data[i].weight *= exp(-alpha_t * class_num / (class_num - 1));
            }
            else // 如果分类错误，则W_i^(t+1) = W_i^(t) * exp(\frac{\alpha_t * K}{(K+1)^2})
            {
                training_data[i].weight *= exp(alpha_t * class_num / ((class_num + 1) * (class_num + 1)));
            }
            sum_w += training_data[i].weight; // 累加权重
        }
        // 对样本权重进行归一化处理
        for (auto& data : training_data)
        {
            data.weight /= sum_w; // 每个样本的权重除以总权重
        }
        return true; // 返回true表示分类器合格
    }
    
public:
    // 把判别器和数据注入，准备进行训练，这些分类器事先是训练好的
    void init(std::vector<judger_tp> vec_judgers, std::vector<weight_data_t> vec_data, 
              const std::vector<int>& vec_expected_results, int i_class_num)
    {
        if (i_class_num < 2)
        {
            printf("Error: class_num must be greater than 1.\n");
            return;
        }
        if (vec_data.size() != vec_expected_results.size())
        {
            printf("Error: vec_data and vec_expected_results must have the same size.\n");
            return;
        }
        classifiers.clear();
        training_data.clear();
        expected_results = vec_expected_results;
        class_num = i_class_num;
        for (auto& j : vec_judgers)
        {
            classifiers.emplace_back(j, 0.0); // 初始化分类器权重为0.0
        }
        for (auto& d : vec_data)
        {
            training_data.emplace_back(d, 1.0/static_cast<double>(training_data.size()));// 初始化数据权重为1.0
        }
    }

    int train()
    {
        if (classifiers.empty() || training_data.empty())
        {
            printf("Error: classifiers or training_data is empty.\n");
            return -1; // 如果分类器或训练数据为空，返回-1表示错误
        }
        int cidx = 0; // 分类器索引
        while (cidx < classifiers.size())
        {
            if (!update_weights(cidx)) // 更新权重，如果分类器不合格，则跳过
            {
                // 删除从cidx开始及以后的所有分类器
                classifiers.erase(classifiers.begin() + cidx, classifiers.end());
                return cidx; // 返回当前分类器数量
            }
            ++cidx; // 分类器合格，继续下一个分类器
        }
        return cidx; // 返回分类器数量
    }

    int predict(const data_t& data)
    {
        std::vector<double> votes(class_num, 0.0); // 初始化投票结果
        double error_base = -1. / (class_num - 1); // 错误基数
        // 对每个分类器进行投票
        for (const auto& classifier : classifiers)
        {
            int result = classifier.value->judge(data); // 获取分类器的判断结果
            if (result >= 0 && result < class_num) // 确保结果在合法范围内
            {
                votes[result] += classifier.weight; // 累加权重
                for (int i = 0; i < class_num; ++i)
                {
                    if (i != result) // 对其他类别进行减权
                    {
                        votes[i] += error_base * classifier.weight;
                    }
                }
            }
        }
        // 找到得票最多的类别
        int max_index = 0;
        double max_value = votes[0];
        for (int i = 1; i < class_num; ++i)
        {
            if (votes[i] > max_value)
            {
                max_value = votes[i];
                max_index = i;
            }
        }
        return max_index; // 返回得票最多的类别索引
    }
};

#endif
