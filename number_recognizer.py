import cv2

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Jersey number recognition will be disabled.")


class NumberRecognizer:
    """
    A class to recognize jersey numbers from player crop images using OCR.
    """

    def __init__(self):
        """
        Initialize the NumberRecognizer.
        """
        if TESSERACT_AVAILABLE:
            print("NumberRecognizer initialized with Tesseract OCR")
        else:
            print("NumberRecognizer initialized (Tesseract not available - will return None)")

    def recognize_number(self, player_crop_image):
        """
        Recognize a jersey number from a cropped player image.

        Args:
            player_crop_image: A cropped image (numpy array) containing a player

        Returns:
            A string containing the recognized number(s), or an empty string if none found
        """
        if not TESSERACT_AVAILABLE:
            return ''

        try:
            # Convert the image to grayscale
            gray = cv2.cvtColor(player_crop_image, cv2.COLOR_BGR2GRAY)

            # Apply binary inverse thresholding
            # This makes text white on black background, which works better for OCR
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Configure Tesseract OCR
            # --psm 6: Assume a single uniform block of text
            # tessedit_char_whitelist=0123456789: Only recognize digits
            custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'

            # Perform OCR
            text = pytesseract.image_to_string(thresh, config=custom_config)

            # Clean the result: strip whitespace and remove newlines
            cleaned_text = text.strip().replace('\n', '').replace(' ', '')

            return cleaned_text

        except Exception as e:
            print(f"Error during OCR: {e}")
            return ''
