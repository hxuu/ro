class UnionFind:
    def __init__(self, headers):
        self.header_to_index = {header: idx for idx, header in enumerate(headers)}
        self.index_to_header = {idx: header for idx, header in enumerate(headers)}
        self.parent = list(range(len(headers)))

    def find(self, header):
        """
            Find the 'representative' of a node label.
        """
        index = self.header_to_index[header]
        parent_index = self.parent[index]

        if parent_index == index:
            return header

        # else, recursively find the 'representative' of the parent
        return self.find(self.index_to_header[parent_index])

    def unite(self, i, j):
        """
            Unite two headers - These nodes will be connected.
        """
        irepr = self.find(i)
        jrepr = self.find(j)

        # make representative to i the same as representative of j
        self.parent[self.header_to_index[irepr]] = self.header_to_index[jrepr]
