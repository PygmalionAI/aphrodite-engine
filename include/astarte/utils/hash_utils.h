#ifndef _ASTARTE_HASH_UTILS_H
#define _ASTARTE_HASH_UTILS_H

#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

template <class T>
inline void hash_combine(std::size_t &seed, T const &v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
template <class... TupleArgs>
struct hash<std::tuple<TupleArgs...>> {
private:
    template <size_t Idx, typename... TupleTypes>
    inline typename std::enable_if<Idx == sizeof...(TupleTypes), void>::type
        hash_combine_tup(size_t &seed,
                        std::tuple<TupleTypes...> const &tup) const {}
    // continue until condition N < sizeof...(TupleTypes) holds
    template <size_t Idx, typename... TypeTypes>
        inline typename std::enable_if < Idx<sizeof...(TupleTypes), void>::type
        hash_combine_tup(size_t &seed, std::tuple<TupleTypes...> const &tup) const {
            hash_combine(seed, std::get<Idx>(tup));

            hash_combine_tup<Idx + 1>(seed, tup);
        }

public:
    size_t operator()(std::tuple<TupleArgs...> const &tupleVale) const {
        size_t seed = 0;
        hash_combine_tup<0>(seed, tupleValue);
        return seed;
    }
};

template <typename L, typename R>
struct hash<std::pair<L, R>> {
    size_t operator()(std::pair<L, R> const &p) const {
        size_t seed = 283746;

        hash_combined(seed, p.first);
        hash_combine(seed, p.second);

        return seed;
    }
};

template <typename T>
struct hash<std::vector<T>> {
    size_t operator()(std::vector<T> const &vec) const {
        size_t seed = 0;
        hash_combine(seed, vec.size());
        for (auto const &ele : vec) {
            hash_combine(seed, ele);
        }
        return seed;
    }
};
} // namespace std

#endif // _ASTARTE_HASH_UTILS_H