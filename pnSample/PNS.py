import torch


def get_node_nld(data, idx_info, minor_class_index):
    """
    Get NLD of all minor nodes

    Input:
        data:               the pyg dataset
        idx_info:           index of all class node
        minor_class_index:  index of minor class, indicated the i th class is minor class,
        for example, class 3 and 4 are minor class, then minor_class_index = [3, 4]

    Output:
        neighbors_list:     NLD of all minor nodes
    """

    edges = data.edge_index

    out_nodes = edges[0].clone()
    in_nodes = edges[1].clone()

    neighbors_list = []
    for i in minor_class_index:
        node_index = idx_info[i]
        node_num = node_index.shape[0]
        neighbors_cls_list = []
        for j in range(node_num):
            temp_neighbors_list = []
            node = node_index[j]
            in_neighbors_index = out_nodes == node
            temp_neighbors_list.append(in_nodes[in_neighbors_index].clone().tolist())
            neighbors_cls_list.append(temp_neighbors_list)

        neighbors_list.append(neighbors_cls_list)

    return neighbors_list

def get_deviation_node(deviation_rate, n_data, minor_class_index, idx_info, data):
        """
        Get deviation node by purity of all minor nodes

        Returns:
            a boolean matrix, indicate whether a node in a minor class is a deviation node.
        """
        neighbors_list = get_node_nld(data, idx_info, minor_class_index)

        # deviation_node = torch.tensor([[] for i in range(len(minor_class_index))], dtype=bool)
        deviation_node = []
        for i in range(len(minor_class_index)):
            cls_pure_node = torch.zeros(size=(n_data[i],), dtype=bool).unsqueeze(1)

            cls = minor_class_index[i]
            class_neighbors_list = neighbors_list[i]
            class_neighbors_label_list = [data.y[node_neighbors].clone().tolist()
                                          for node_neighbors in class_neighbors_list]

            # pure_index = [index
            #               for index, node_neighbors_label in enumerate(class_neighbors_label_list)
            #               if (node_neighbors_label.count(cls) / len(node_neighbors_label)) > deviation_rate]
            pure_index = [index
                          for index, node_neighbors_label in enumerate(class_neighbors_label_list)
                          if len(node_neighbors_label) != 0 and (
                                      node_neighbors_label.count(cls) / len(node_neighbors_label)) > deviation_rate]

            cls_pure_node[pure_index] = True
            cls_deviation_node = ~cls_pure_node

            deviation_node.append(cls_deviation_node)

        return deviation_node


def get_pure_node(deviation_rate, n_data, minor_class_index, idx_info, data):
        """
        Get pure node by purity of all minor nodes

        Returns:
            a boolean matrix, indicate whether a node in a minor class is a pure node.
        """
        neighbors_list = get_node_nld(data, idx_info, minor_class_index)

        pure_node = []
        for i in range(len(minor_class_index)):
            cls_pure_node = torch.zeros(size=(n_data[i],), dtype=bool).unsqueeze(1)

            cls = minor_class_index[i]
            class_neighbors_list = neighbors_list[i]
            class_neighbors_label_list = [data.y[node_neighbors].clone().tolist()
                                          for node_neighbors in class_neighbors_list]

            pure_index = [index
                          for index, node_neighbors_label in enumerate(class_neighbors_label_list)
                          if len(node_neighbors_label) != 0 and (node_neighbors_label.count(cls) / len(node_neighbors_label)) > deviation_rate]


            cls_pure_node[pure_index] = True

            pure_node.append(cls_pure_node)

        return pure_node

