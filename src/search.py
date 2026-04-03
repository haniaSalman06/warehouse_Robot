"""Search algorithms for warehouse robot problem."""
import heapq
import time
import mlflow

class UniformCostSearch:
    """Uniform Cost Search implementation."""
    
    def __init__(self, problem):
        self.problem = problem
        self.nodes_expanded = 0
        self.max_frontier_size = 0

    def solve(self, initial_state, log_to_mlflow=False):
        """Solve using UCS."""
        start_time = time.time()
        
        # Priority queue: (cost, state, path)
        frontier = [(0, initial_state, [initial_state.position])]
        visited = set()

        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            cost, state, path = heapq.heappop(frontier)

            # Goal test
            if state.is_goal(self.problem):
                elapsed_time = time.time() - start_time
                if log_to_mlflow:
                    self._log_experiment(cost, elapsed_time)
                return path, cost, self.nodes_expanded, elapsed_time

            # Skip if visited
            if state in visited:
                continue
            visited.add(state)
            self.nodes_expanded += 1

            # Explore neighbors
            for neighbor in state.get_neighbors(self.problem):
                if neighbor not in visited:
                    new_cost = cost + 1
                    new_path = path + [neighbor.position]
                    heapq.heappush(frontier, (new_cost, neighbor, new_path))

        return None, float('inf'), self.nodes_expanded, 0

    def _log_experiment(self, cost, time_taken):
        """Log experiment metrics to MLflow."""
        mlflow.log_param("algorithm", "UCS")
        mlflow.log_metric("total_cost", cost)
        mlflow.log_metric("nodes_expanded", self.nodes_expanded)
        mlflow.log_metric("max_frontier_size", self.max_frontier_size)
        mlflow.log_metric("time_seconds", time_taken)