# Copyright (C) 2025 Hanyu Liu
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.


class LinearSearch(object):
    def __init__(self,
                 right_x: float = 1.0,
                 max_iters: int = 999):
        """
        Args:
            right_x: Initial right boundary for linesearch 
            max_iters: Maximum number of linesearch iterations.
        """
        self.initial_right_x = right_x
        self.max_iters: int = max_iters

        # State variable
        self.curr_x: float = 0.0
        self.prev_x: float = 0.0
        self.prev_f: float = 0.0
        self.prev_g: float = 0.0

        # Boundary information
        self.left_x: float = 0.0
        self.left_f: float = 0.0
        self.left_g: float = 0.0

        self.right_x: float = 0.0
        self.right_f: float = 0.0
        self.right_g: float = 0.0

        # Iteration count
        self.iter: int = 0
        self.reset(initial_step=right_x)
    

    def reset(self,
              initial_step: float = 1.0):
        """Reset line search state for a new search"""
        self.curr_x = 0.0
        self.left_x = 0.0
        self.right_x = initial_step
        self.right_f = float('inf')
        self.right_g = float('inf')
        self.iter = 0

    
    def reduce_step(self,
                    ratio: float = 0.25):
        self.curr_x = self.prev_x + ratio * (self.curr_x - self.prev_x)

    
    def iterate(self,
                curr_f: float,
                curr_g: float) -> bool:
        """
        Perform one iteration of linesearch

        Args:
            curr_f: Current function value at curr_x
            curr_g: Current gradient projected on search direction at curr_x (p_dot_g)
        
        Returns:
            True if line search should continue, False if search fails
        """
        # The very first iteration
        if self.curr_x == 0.0:
            self.left_x = self.curr_x
            self.left_f = curr_f
            self.left_g = curr_g

            self.prev_x = self.curr_x
            self.prev_f = curr_f
            self.prev_g = curr_g

            self.curr_x = self.right_x
            
            if (curr_g > 0):
                print("Warning: Linesearch with increasing function!")
                return False
            return True
        
        # Now prev_x, prev_f, prev_g are set
        if self.prev_x == self.curr_x:
            print("Warning: prev_x == curr_x")
            return False
        
        # Set default right boundary
        if (self.curr_x == self.right_x):
            pass
