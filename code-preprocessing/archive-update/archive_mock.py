# -*- coding: utf-8 -*-

class Archive:
    """A mock of the Archive (to be replaced by the real thing when it starts
       working).
    """    
    
    def __init__(self, suite_name, function, dimension, instance):
        self.solutions = []
        self.current_solution = 0
        self.limit_solutions = 100
                            
    def add_solution(self, f1, f2, text):
        if (self.limit_solutions > 0) and (len(self.solutions) >= self.limit_solutions):
            return False
            
        self.solutions.append([f1, f2, text])
        return True
        
    def get_next_solution_text(self):
        if self.current_solution >= self.number_of_solutions:
            return None
            
        self.current_solution += 1
        
        return self.solutions[self.current_solution - 1][2]
                
    @property
    def number_of_solutions(self):
        return (len(self.solutions))
        
    @property
    def hypervolume(self):
        return 0.42
    
    