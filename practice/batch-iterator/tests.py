from practice_check import check, fixture, fixtures, raises, true


def same(actual, expected, what):
    true(actual == expected, f"{what} is not as expected", expected, actual)


@check("examples", "7 items in batches of 3", "The last batch holds the one item that is left.")
def example_basic(sub):
    same(list(sub.batches(range(7), 3)), [[0, 1, 2], [3, 4, 5], [6]], "the batches")


@check("examples", "drop_last removes the short batch", "With drop_last True, the batch [6] has fewer than 3 items, so it is not returned.")
def example_drop_last(sub):
    same(list(sub.batches(range(7), 3, drop_last=True)), [[0, 1, 2], [3, 4, 5]], "the batches")


@check("examples", "an empty input gives no batches", "There is nothing to group, so the iterator ends at once.")
def example_empty(sub):
    same(list(sub.batches([], 3)), [], "the batches")


@check("properties", "the batches join back to the input", "Without drop_last, no item is lost, repeated or moved.")
def join_back(sub):
    for n in (1, 5, 12, 13):
        for size in (1, 2, 4, 20):
            joined = [x for batch in sub.batches(list(range(n)), size) for x in batch]
            same(joined, list(range(n)), f"the joined batches for n={n}, batch_size={size}")


@check("properties", "every batch is full except maybe the last", "Each batch has exactly batch_size items. Only the last batch can be shorter.")
def batch_lengths(sub):
    out = list(sub.batches(range(23), 5))
    same([len(b) for b in out], [5, 5, 5, 5, 3], "the batch lengths")


@check("properties", "the result is an iterator", "Call next on the result to get one batch. A plain list is not an iterator.")
def returns_iterator(sub):
    result = sub.batches(range(6), 2)
    true(hasattr(result, "__next__"), "batches must return an iterator", "an object with __next__", type(result).__name__)
    same(next(result), [0, 1], "the first batch")


@check("properties", "items are read lazily", "The function takes only the items that the next batch needs. It must not read the whole input first.")
def lazy(sub):
    log = []

    def source():
        for i in range(100):
            log.append(i)
            yield i

    it = sub.batches(source(), 4)
    first = next(it)
    same(first, [0, 1, 2, 3], "the first batch")
    true(len(log) <= 4, "the function read more items than the first batch needs", "at most 4 items read", len(log))


@check("properties", "a generator that can be read only once works", "A generator is empty after one pass, so the function must read it one time only.")
def one_pass_generator(sub):
    gen = (i * i for i in range(7))
    same(list(sub.batches(gen, 3)), [[0, 1, 4], [9, 16, 25], [36]], "the batches of the generator")


@check("properties", "earlier batches are not changed later", "Each batch must be a new list. If the code reuses one list, the batches that you kept change.")
def separate_lists(sub):
    out = list(sub.batches(range(6), 2))
    same(out, [[0, 1], [2, 3], [4, 5]], "the kept batches")
    true(len({id(b) for b in out}) == 3, "the batches share one list object", "3 different lists", len({id(b) for b in out}))


@check("properties", "the input list is not changed", "A function that sorts or empties its input breaks the code that calls it.")
def input_unchanged(sub):
    data = [5, 3, 9, 1, 7]
    list(sub.batches(data, 2))
    same(data, [5, 3, 9, 1, 7], "the input list")


@check("edge", "batch size larger than the input", "One short batch holds all items. With drop_last it is removed, so nothing is returned.")
def big_batch(sub):
    same(list(sub.batches([1, 2, 3], 10)), [[1, 2, 3]], "the batches")
    same(list(sub.batches([1, 2, 3], 10, drop_last=True)), [], "the batches with drop_last")


@check("edge", "batch size 1 and an exact multiple", "With batch size 1 every item is its own batch. When the length is a multiple of the batch size, drop_last changes nothing.")
def exact_multiple(sub):
    same(list(sub.batches([1, 2, 3], 1)), [[1], [2], [3]], "the batches of size 1")
    same(list(sub.batches(range(6), 3, drop_last=True)), [[0, 1, 2], [3, 4, 5]], "the batches with drop_last")


@check("edge", "a batch size below 1 raises ValueError", "Batch size 0 or a negative number has no meaning. The function must raise ValueError at the call.")
def bad_size(sub):
    raises(ValueError, sub.batches, [1, 2, 3], 0)
    raises(ValueError, sub.batches, [1, 2, 3], -2)


@fixture("cases")
def cases(ref):
    for n, size, drop in [(0, 3, False), (1, 1, False), (10, 3, False), (10, 3, True), (12, 4, True), (9, 20, False), (9, 20, True), (31, 5, False)]:
        yield {"n": n, "size": size, "drop": drop}, [list(b) for b in ref.batches(range(n), size, drop)]


@check("reference", "batches match the reference", "Your batches equal the reference batches for many sizes, with and without drop_last.")
def reference_batches(sub):
    for case, expected in fixtures("cases"):
        same(list(sub.batches(range(case["n"]), case["size"], case["drop"])), expected, f"the batches for n={case['n']}, size={case['size']}, drop_last={case['drop']}")
