class PointCloudWrapper:
    def calculate_focals(self, boxes, classes, distances):
        vertical = []
        horizontal = []

        for box, object_class, distance in zip(boxes, classes, distances):
            if object_class in self.disnet.class_sizes.keys() and distance:
                real_height = self.disnet.class_sizes[object_class]['size'][0]
                real_width = self.disnet.class_sizes[object_class]['size'][1]

                # TODO -check if not other way around
                calc_height = abs(box[2] - box[0])
                calc_width = abs(box[3] - box[1])

                vertical.append((calc_height * distance) / (
                            real_height / 100) * 0.265)  # convert distance units to meters and then pixels to mm
                horizontal.append((calc_width * distance) / (
                            real_width / 100) * 0.265)  # convert distance units to meters and then pixels to mm
            else:
                vertical.append(None)
                horizontal.append(None)

        return vertical, horizontal

    def _process_regression(self, inv_rel_depth, boxes, distances):
        if self.use_midas and self.use_disnet:
            fit_status, distance_frame = self.distance_regressor.predict(inv_rel_depth, boxes, distances)
            # TODO - add logging of a distance frame
        else:
            fit_status = False

        return fit_status