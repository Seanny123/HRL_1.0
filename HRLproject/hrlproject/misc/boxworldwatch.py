from java.awt import Color
from java.lang import Class, NoSuchFieldException

from timeview.watches.watchtemplate import WatchTemplate
from timeview import components

class BoxWorldWatch(WatchTemplate):
    """Watch for the BoxWorldEnvironment (for use in interactivemode display)."""
    
    def check(self, obj):
        """Returns True if the object is a thing this watch is associated with."""
        
        return hasattr(obj, "get_boxes")
    
    def display_boxes(self, obj):
        """Returns the data to be displayed by components.Boxes."""
        
        return [[self.color_translation(b[0]), b[1], b[2]] for b in obj.get_boxes()] 
    
    def color_translation(self, data):
        """Map data returned from obj.get_boxes to a java Color object."""
        
        # could be an RGB tuple
        if isinstance(data,(tuple,list)):
            return Color(data[0],data[1],data[2])
        
        # or a string (e.g., "red")
        if isinstance(data,basestring):
            try:
                field = Class.forName("java.awt.Color").getField(data)
                return field.get(None)
            except NoSuchFieldException:
                pass
            
        # special case for specifying lines rather than boxes
        if data == "-": 
            return data
        
        print "Unknown data in boxworldwatch color_translation: " + str(data)
        return None
        
    def views(self, obj):
        r = [("display boxes", components.Boxes, dict(func=self.display_boxes, label=obj.name))]
        return r