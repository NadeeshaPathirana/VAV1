# # # Source - https://stackoverflow.com/a
# # # Posted by furas, modified by community. See post 'Timeline' for change history
# # # Retrieved 2025-11-20, License - CC BY-SA 4.0
# #
# # import webview
# # import time
# #
# # def destroy(window):
# #     # show the window for a few seconds before destroying it:
# #     time.sleep(5)
# #     print("I am talking now")
# #     print('Destroying window..')
# #     window.destroy()
# #     print('Destroyed!')
# #
# # if __name__ == '__main__':
# #     window = webview.create_window('Destroy Window Example', 'https://media.tenor.com/ZViDCL9tx_QAAAAi/set-diet-sound-bars.gif')
# #     webview.start(destroy, window)
# #     print('Window is destroyed')
#
#
# import sounddevice as sd
#
# import numpy as np
#
# import pyautogui
#
# frequency_threshold = 1000
#
# while True:
#
# data = sd.rec(1024, 44100, channels=2)
#
# frequencies, times, spectrogram = stft(data, 44100, nperseg=1024)
#
# max_frequency = np.abs(frequencies[np.argmax(spectrogram)])
#
# if max_frequency > frequency_threshold:
#
# # Perform actions with PyAutoGUI here
#
# else:
#
# # Stop actions here