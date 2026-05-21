class CommunicationModule:

    def __init__(self, neighbors, max_neighbours=None, comm_msg_size=4):
        self.neighbors = neighbors
        self.messages = {}
        self.comm_msg_size = comm_msg_size
        if max_neighbours is not None:
            self.max_neighbours = max_neighbours
        else:
            self.max_neighbours = max(len(v) for v in neighbors.values()) if neighbors else 0

    def update_message(self, tl_id, message):
        self.messages[tl_id] = message

    def get_neighbor_messages(self, tl_id):
        output = []

        for neighbor_id in self.neighbors.get(tl_id, []):
            if neighbor_id in self.messages:
                output.extend(self.messages[neighbor_id])
            else:
                output.extend([0] * self.comm_msg_size)

        # Pad so every agent sees the same number of neighbour slots
        actual = len(self.neighbors.get(tl_id, []))
        output.extend([0] * ((self.max_neighbours - actual) * self.comm_msg_size))

        return output