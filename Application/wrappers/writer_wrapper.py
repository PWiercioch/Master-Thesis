# TODO - handle variables
class WriterWrapper:
    def _write(self, out, video, fit_status, boxes, classes, scores, distances, focal_v, focal_h, comment):
        ### Writing video
        if self.config["record_annotated"]:
            if self.config["record_annotated"]:
                out.write(video.get_frame("annotated"))
            else:
                out.write(video.get_frame("raw"))
        else:
            if self.use_midas:
                out.write(video.get_frame("alpha_record"))
            else:
                out.write(video.get_frame("raw"))

        ### Writing log
        if fit_status:
            coefs = self.distance_regressor.regression_model.get_coeffs()
        else:
            coefs = None

        out.log(boxes, classes, scores, distances, focal_v, focal_h, coefs, comment)