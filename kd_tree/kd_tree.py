from collections import deque
k=0

class KDTreeNode:
    def __init__(self,point):
        self.point=point
        self.left=None
        self.right=None

def load_data():
    data=[]
    with open("points3.txt", "r") as file:
        for line in file:
            if line.strip():
                line=line.strip()
                line=line.strip("()")
                point=line.split(",")
                data.append(list(map(int,point)))
    return data

def insert_point(point,node,depth=0):
    if node==None:
        return KDTreeNode(point)

    cutdim=depth%k

    if point[cutdim]<=node.point[cutdim]:
        node.left=insert_point(point,node.left,depth+1)
    else:
        node.right=insert_point(point,node.right,depth+1)
    return node

def build_kd_tree(tree,data,depth=0):
    if len(data)==0:
        return tree
    
    cutdim=depth%k
    median_index=(len(data)-1)//2
    data=sorted(data,key=lambda x: x[cutdim])
    left_data= data[:median_index]
    right_data= data[median_index+1:]
    median_point=data[median_index]
    print(data)
    print(f"leftlist: {left_data}, median: {median_point}, rightlist: {right_data}")
    
    tree=insert_point(median_point,tree)

    tree=build_kd_tree(tree,left_data,depth+1)
    tree=build_kd_tree(tree,right_data,depth+1)
    return tree

def delete_node(root, point, depth=0):
    if root is None:
        return None

    current_dim = depth % k

    if root.point == point:
        if root.right is not None:
            min_node = find_min(root.right, current_dim, depth+1)
            root.point = min_node.point
            #print(f"min node is {min_node.point}")
            root.right = delete_node(root.right, min_node.point, depth + 1)
        elif root.left is not None:
            min_node = find_min(root.left, current_dim, depth+1)
            root.point = min_node.point
            root.right = delete_node(root.left, min_node.point, depth + 1)  
            root.left = None
        else:
            return None 

    elif point[current_dim] < root.point[current_dim]:
        root.left = delete_node(root.left, point, depth + 1)
    else:
        root.right = delete_node(root.right, point, depth + 1)

    return root

def find_min(root, dim, depth=0):
    if root is None:
        return None
    current_dim = depth % k

    if current_dim == dim:
        if root.left is None:
            #print(f"being returned: {root.point}")
            return root
        return find_min(root.left, dim, depth + 1)
    
    left_min = find_min(root.left, dim, depth + 1)
    right_min = find_min(root.right, dim, depth + 1)

    return min(
        (node for node in [root, left_min, right_min] if node is not None),
        key=lambda x: x.point[dim]
    )

def distSquared(target, point):
    return sum((a - b) ** 2 for a, b in zip(target, point))

def closest(target, temp, node):
    if temp is None:
        return node
    
    dist = distSquared(target, temp.point)
    dist_other = distSquared(target, node.point)

    return temp if dist < dist_other else node


def nearest_neighbor(node, target, depth=0):
    if node is None:
        return None
    
    next_branch, other_branch = None, None
    if target[depth%k]<node.point[depth%k]:
        next_branch, other_branch = node.left, node.right
    else:
        next_branch, other_branch = node.right, node.left
    
    temp = nearest_neighbor(next_branch,target,depth+1)
    best = closest(target,temp,node)

    r = distSquared(target,best.point)
    r_prime=target[depth%k]-node.point[depth%k]

    if r>=r_prime*r_prime:
        temp = nearest_neighbor(other_branch, target, depth+1)
        best = closest(target, temp, best)
    return best

def print_tree(root, space="", side="root"):
    if root is None:
        return
    print(space, root.point, f"({side})")
    print_tree(root.left, space + "  ", "left")
    print_tree(root.right, space + "  ", "right")

if __name__=='__main__':
    data=load_data()
    kd_tree=None
    k=len(data[0])
    print(data)
    kd_tree=build_kd_tree(kd_tree,data)
    print_tree(kd_tree,"","root")

    #kd_tree=delete_node(kd_tree,[5,8])
    #kd_tree=deleteNode(kd_tree,[4,4])
    print()

    #print_tree(kd_tree,"","root")

    # closest_neighbor=nearest_neighbor(kd_tree,[7,2])
    # print(closest_neighbor.point)

