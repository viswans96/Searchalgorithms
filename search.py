# coding=utf-8

import heapq
import os
import pickle
import math
from collections import deque

class PriorityQueue(object):
  
    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.index = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        popped = heapq.heappop(self.queue)
        return (popped[0], popped[-1])
        #raise NotImplementedError

    def remove(self, node):
               
        for i in range(len(self.queue)):
            if self.queue[i][-1][0] == node[-1][0]:
                del self.queue[i]
                break
        heapq.heapify(self.queue)
        #return (removed[0], removed[-1])        
        #raise NotImplementedError
        
    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
       
        heapq.heappush(self.queue, (node[0], self.index, node[-1]))
        self.index += 1
        # raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]
        
    def contains_tuple(self, key):
        self.index = 0
        return key in [n[0] for v,b,n in self.queue]

    def get_item(self, key):
        self.index = 0
        return [n for v,b,n in self.queue if n[0] == key]
        
    def get_priority(self, key):
        self.index = 0
        return [v for v,b,n in self.queue if n[0] == key]    
        
    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """
        firstitem = self.queue[0]
        return (firstitem[0], firstitem[-1]) 


def breadth_first_search(graph, start, goal):
    
    if start == goal:
        return []
    queue = deque([start])
    explored  = {}
    explored[start] = None
    
    while queue:
        current_node = queue.popleft()  
        if current_node == goal:
            path = get_path(start, goal, explored)
            return path
        
        neighbours = graph[current_node]
        neighbours_sorted = sorted(neighbours)
    
        for neighbour in neighbours_sorted:
            if neighbour not in explored and neighbour not in queue:
                if neighbour == goal:
                    explored[neighbour] = current_node
                    path = get_path(start,goal,explored)
                    return path
                queue.append(neighbour)
                explored[neighbour] = current_node 
     
    #raise NotImplementedError

def get_path(start, goal, explored):
    current = goal 
    return_path = []
    while current != start: 
        return_path.append(current)
        current = explored[current]
    return_path.append(start) 
    path = return_path[::-1] 
    return path  

    
def update_current_node(priority_queue, current_cost, node):
    node_priority = node[0]
    node_to_pass = node[1][0]
    get_node = priority_queue.get_priority(node_to_pass)
    if node_priority < get_node[0]:
        priority_queue.remove(node)
        priority_queue.append(node)
        current_cost[node_to_pass] = node_priority    


def add_paths(path1, path2, is_forward=True):
    if is_forward:
        path2r = path2[::-1]
        path1.extend(path2r)
        return path1
    else:
        path1r = path1[::-1]
        path2.extend(path1r)
        return path2
        

def currentpath_cost(graph, path):
    cost = 0
    path_len = len(path)
    if path_len == 1:
        return cost

    for i in range(path_len):
        if i + 1 <= path_len - 1:
            cost += graph.get_edge_weight(path[i], path[i+1])
    return cost        


def uniform_cost_search(graph, start, goal):
    if start == goal:
        return []
    priority_queue = PriorityQueue()
    priority_queue.append((0, start))
    
    explored  = {}
    explored[start] = None
    
    cost = {}
    cost[start] = 0
    
    while priority_queue:
        priority_node = priority_queue.pop()
        current_node = priority_node[1]
        if current_node == goal:
            break
        neighbours = graph[current_node]
        for neighbour in neighbours:
            neighbour_cost = cost[current_node] + graph.get_edge_weight(current_node, neighbour) 
            if neighbour not in cost or neighbour_cost < cost[neighbour]:
                priority_queue.append((neighbour_cost, neighbour))
                cost[neighbour] = neighbour_cost
                explored[neighbour] = current_node
   
    path = get_path(start,goal,explored)
    return path    
    #raise NotImplementedError


def null_heuristic(graph, v, goal):
   
    return 0


def euclidean_dist_heuristic(graph, v, goal):
    
    # TODO: finish this function!
    current_pos = graph.nodes[v]['pos']
    goal_pos = graph.nodes[goal]['pos']

    return math.sqrt(math.pow(current_pos[0] - goal_pos[0], 2) + math.pow(current_pos[1] - goal_pos[1], 2))
    # raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    
    # TODO: finish this function!
    if start == goal:
        return []
    priority_queue = PriorityQueue()
    priority_queue.append((0, start))
    
    explored  = {}
    explored[start] = None
    
    cost = {}
    cost[start] = 0
    
    while priority_queue:
        priority_node = priority_queue.pop()
        current_node = priority_node[1]
        if current_node == goal:
            break
        neighbours = graph[current_node]
        for neighbour in neighbours:
            neighbour_cost = cost[current_node] + graph.get_edge_weight(current_node, neighbour) 
            if neighbour not in cost or neighbour_cost < cost[neighbour]:
                priority = neighbour_cost + heuristic(graph, neighbour, goal)
                priority_queue.append((priority, neighbour))
                cost[neighbour] = neighbour_cost
                explored[neighbour] = current_node
   
    path = get_path(start,goal,explored)
    return path  
    # raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    if start == goal:
        return []

    forwards_explored = {}
    backwards_explored = {}

    cost_forwards = {}
    cost_backwards = {}

    queue_forwards = PriorityQueue()
    queue_backwards = PriorityQueue()
    
    queue_forwards.append((0, (start, [start])))
    queue_backwards.append((0, (goal, [goal])))

    cost_forwards[start] = 0
    cost_backwards[goal] = 0

    result_paths = PriorityQueue()
    best_cost = None

    while queue_forwards and queue_backwards:

        if best_cost:
            top_forwards = queue_forwards.top()
            top_backwards = queue_backwards.top()
            if cost_forwards[top_forwards[1][0]] + cost_backwards[top_backwards[1][0]] >= best_cost:
                break

        if queue_forwards.top()[0] <= queue_backwards.top()[0]:
            queue_to_expand = queue_forwards
            explored = forwards_explored
            other_explored = backwards_explored
            current_cost = cost_forwards
            is_forward = True
        else:
            queue_to_expand = queue_backwards
            explored = backwards_explored
            other_explored = forwards_explored
            current_cost = cost_backwards
            is_forward = False

        priority_node = queue_to_expand.pop()
        current_node = priority_node[1][0]
        current_node_path = priority_node[1][1]

        explored[current_node] = current_node_path
        neighbours = graph[current_node]
        for neighbour in neighbours:
            if not neighbour in explored:
                new_path = list(current_node_path)
                new_path.append(neighbour)
                new_path_cost = current_cost[current_node] + graph.get_edge_weight(current_node, neighbour) 
                neighbour_node = (new_path_cost, (neighbour, new_path))
                if not queue_to_expand.contains_tuple(neighbour):
                    queue_to_expand.append(neighbour_node)
                    current_cost[neighbour] = new_path_cost
                else:
                    update_current_node(queue_to_expand, current_cost, neighbour_node)

                if neighbour in other_explored:
                    current_path = list(current_node_path)
                    other_path = list(other_explored[neighbour])
                    result_path = add_paths(current_path, other_path, is_forward)
                    result_cost = currentpath_cost(graph, result_path)
                    result_paths.append([result_cost, result_path])

                    if (best_cost and best_cost > result_cost) or not best_cost:
                        best_cost = result_cost

    return result_paths.pop()[1]
    # raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
   
    if start == goal:
        return []

    forwards_explored = {}
    backwards_explored = {}

    cost_forwards = {}
    cost_backwards = {}

    queue_forwards = PriorityQueue()
    queue_backwards = PriorityQueue()
    
    queue_forwards.append((0, (start, [start])))
    queue_backwards.append((0, (goal, [goal])))

    cost_forwards[start] = 0
    cost_backwards[goal] = 0

    result_paths = PriorityQueue()
    best_cost = None

    while queue_forwards and queue_backwards:

        if best_cost:
            top_forwards = queue_forwards.top()
            top_backwards = queue_backwards.top()
            if cost_forwards[top_forwards[1][0]] + cost_backwards[top_backwards[1][0]] >= best_cost:
                break

        if queue_forwards.top()[0] <= queue_backwards.top()[0]:
            queue_to_expand = queue_forwards
            explored = forwards_explored
            other_explored = backwards_explored
            current_cost = cost_forwards
            is_forward = True
            current_goal = goal
        else:
            queue_to_expand = queue_backwards
            explored = backwards_explored
            other_explored = forwards_explored
            current_cost = cost_backwards
            is_forward = False
            current_goal = start

        priority_node = queue_to_expand.pop()
        current_node = priority_node[1][0]
        current_node_path = priority_node[1][1]

        explored[current_node] = current_node_path
        neighbours = graph[current_node]
        for neighbour in neighbours:
            if not neighbour in explored:
                new_path = list(current_node_path)
                new_path.append(neighbour)
                new_path_cost = current_cost[current_node] + graph.get_edge_weight(current_node, neighbour) 
                priority = new_path_cost + heuristic(graph, neighbour, current_goal) 
                neighbour_node = (priority, (neighbour, new_path))
                if not queue_to_expand.contains_tuple(neighbour):
                    queue_to_expand.append(neighbour_node)
                    current_cost[neighbour] = new_path_cost
                else:
                    update_current_node(queue_to_expand, current_cost, neighbour_node)

                if neighbour in other_explored:
                    current_path = list(current_node_path)
                    other_path = list(other_explored[neighbour])
                    result_path = add_paths(current_path, other_path, is_forward)
                    result_cost = currentpath_cost(graph, result_path)
                    result_paths.append([result_cost, result_path])

                    if (best_cost and best_cost > result_cost) or not best_cost:
                        best_cost = result_cost

    return result_paths.pop()[1] 
    # TODO: finish this function!
    # raise NotImplementedError


def tridirectional_search(graph, goals):
    
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    # TODO: finish this function
    raise NotImplementedError

