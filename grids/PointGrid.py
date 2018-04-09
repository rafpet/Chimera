


import Rhino.Geometry as rg


class Grid(object):
    
    def __init__(self,_origin,_x_res,_y_res,_spacing):
        
        self.origin = _origin
        self.x_res = _x_res
        self.y_res = _y_res
        self.spacing = _spacing
        
        self.data_structure = []
        self.initialise()
        
    def initialise(self):
        
        for i in range(self.x_res):
            column = []
            for j in range(self.y_res):
                
                new_point = self.origin + rg.Point3d(i*self.spacing,j*self.spacing,0)
                column.append(new_point)
                
            self.data_structure.append(column)
        
    def get_points_as_list(self):
        points = []
        for i in range(len(self.data_structure)):
            for j in range(len(self.data_structure[i])):
                points.append(self.data_structure[i][j])
        return points
        
    def get_indices_as_list(self):
        indices = []
        for i in range(len(self.data_structure)):
            for j in range(len(self.data_structure[i])):
                index = "%i.%i"%(i,j)
                indices.append(index)
        return indices
        
    def compute_edges(self):
        
        lines = []
        
        for i in range(len(self.data_structure)-1):
            for j in range(len(self.data_structure[i])):
                
                p_A = self.data_structure[i][j]
                p_B = self.data_structure[i+1][j]
                
                lines.append(rg.Line(p_A,p_B))
                
            p_A = self.data_structure[-2][j]
            p_B = self.data_structure[-1][j]
            
            lines.append(rg.Line(p_A,p_B))
            
            
        for i in range(len(self.data_structure)):
            for j in range(len(self.data_structure[i])-1):
                
                p_A = self.data_structure[i][j]
                p_B = self.data_structure[i][j+1]
                
                lines.append(rg.Line(p_A,p_B))
                
            p_A = self.data_structure[-2][j]
            p_B = self.data_structure[-1][j]
            
            lines.append(rg.Line(p_A,p_B))
            
        return lines
        






