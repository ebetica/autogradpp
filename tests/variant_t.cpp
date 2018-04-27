#include "test.h"

Variant identity(const Variant& x) {
    return x;
}

CASE("variant/types") {
    {
        auto x = Var(at::CPU(at::kFloat).randn({5, 10}));
        auto y = identity(x);
        EXPECT(y.get().size(0) == 5);
        EXPECT(y.get().size(1) == 10);
    }
    {
        auto x = std::string("Hello!");
        auto y = identity(x);
        EXPECT(y.getString() == "Hello!");
    }
    {
        auto x = 1.f;
        auto y = identity(x);
        EXPECT(y.getFloat() == 1.f);
    }
    {
        auto x = 2.;
        auto y = identity(x);
        EXPECT(y.getDouble() == 2.);
    }
    {
        auto x = false;
        auto y = identity(x);
        EXPECT(!y.getBool());
    }
    {
        int32_t x = 2;
        auto y = identity(x);
        EXPECT(y.getInt32() == 2);
    }
    {
        int64_t x = 5;
        auto y = identity(x);
        EXPECT(y.getInt64() == 5);
    }
    {
        auto vec = std::vector<Variant>();
        for (auto i = 5; i < 10; i++) {
            vec.push_back(Var(at::CPU(at::kFloat).randn({i})));
        }
        auto y = identity(vec);
        for (auto i = 5; i < 10; i++) {
            EXPECT(y.getList()[i - 5].get().size(0) == i);
        }
    }
    {
        auto map = std::unordered_map<std::string, Variant>();
        for (auto i = 5; i < 10; i++) {
            map[std::to_string(i)] = Var(at::CPU(at::kFloat).randn({i}));
        }
        auto y = identity(map);
        for (auto i = 5; i < 10; i++) {
            EXPECT(y.getDict()[std::to_string(i)].get().size(0) == i);
        }
    }
}
