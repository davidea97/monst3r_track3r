
class ObjectTracker:
    def __init__(self, all_3d_obj_pts, all_obj_msks):
        self.all_3d_obj_pts = all_3d_obj_pts
        self.all_obj_msks = all_obj_msks

    def _get_object_quantity(self):
        return len(self.all_3d_obj_pts)

    def _get_3d_object_pts(self, obj_id):
        return self.all_3d_obj_pts[obj_id]
    
    def _get_object_mask(self, obj_id):
        return self.all_obj_msks[obj_id]
    
    def _get_all_3d_object_pts(self):
        return self.all_3d_obj_pts
    
    def _get_all_object_masks(self):
        return self.all_obj_msks
    
