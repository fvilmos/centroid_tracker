import cv2

class Cascade():
    
    def __init__(self,data_file):
        """[summary]

        Args:
            data_file (str): path to the cascade file
        """

        self.cc = cv2.CascadeClassifier()
        self.cc.load(data_file)
        self.objects = []
        self.scale = 1.1
        self.neighbor = 0

    def set_parameters(self, scale=1.1, neighbor=0):
        """
        Set detection parameters
        Args:
            scale (float, optional): Scale factor. Defaults to 1.1.
            neighbor (int, optional): Number of neighbors for detections. Defaults to 0.
        """
        self.scale = scale
        self.neighbor = neighbor

    def get_detections(self, img):
        """
        makes cascade detections

        Args:
            img (image): input - RGB image
            returns a list of (x,y,w,h) for objects detected
        """
        img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_g = cv2.equalizeHist(img_g)

        detections = self.cc.detectMultiScale(img_g,self.scale,self.neighbor, flags=cv2.CASCADE_DO_CANNY_PRUNING)

        return detections

    def display(self, img):
        """
        display on an image the detections

        Args:
            img (image): color RGB image
        """
        detections = self.get_detections(img)

        for (x,y,w,h) in detections:
            cv2.rectangle(img,(x,y),(x+w,y+h),[0,255,0],2)
        
        return img
