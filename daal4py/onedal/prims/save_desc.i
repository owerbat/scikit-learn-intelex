class computer
{
public:
    virtual ~computer()    = default;
    virtual void compute() = 0;
};

template <typename... Args, typename... Ts>
class tcomputer : public computer
{
public:
    tcomputer(Ts &&... args) : input_args_(args) {}

    void compute() override
    {
        linear_kernel::descriptor<Args...> descriptor {};
        python::compute2(descriptor, input_args_)
    }

private:
    std::tuple<Ts...> input_args_;
};

template <typename... Args, typename... Ts>
auto make_computer(Ts &&... args)
{
    return std::shared_ptr<tcomputer<Args..., Ts...> >(new tcomputer<Args..., , Ts...>(std::forward<Ts>(args)...));
}

class matcher
{
public:
    using Key   = std::string;
    using Value = std::shared_ptr<computer>;

    matcher(const std::unordered_map<Key, Value> & map) : _map(map) {}

    void compute(const Key & params)
    {
        auto c = _map.at(params);
        c->compute();
    }

private:
    std::unordered_map<Key, Value> _map;
};