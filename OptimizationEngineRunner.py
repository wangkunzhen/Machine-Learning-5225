from OptimizationEngine import OptimizationEngine


class ExecutionEngineStub:
    def cost_T(self, order_book, mid_spread, inventory):
        return mid_spread * inventory

    def cost_other(self, messaage_book, order_book, inventory, mid_spread, action):
        return range(0, inventory+1), [a * mid_spread for a in range(0, inventory + 1)]


exe_engine = ExecutionEngineStub()
message_book = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
order_book = [[0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 2, 0]]
opt_engine = OptimizationEngine(order_book, message_book, 0, 2, 5, 1, 10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
strategy = opt_engine.compute_optimal_solution(10, exe_engine)
print(strategy)