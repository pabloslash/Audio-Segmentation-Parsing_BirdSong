path_to_save = "/home/aparna/"

spectrogram_specs = {
	"sr": 30000, #sampling rate
	"n_fft": 256, #no. of fft points
	"window": 256, 
	"stride": 64,
	"fft_center": True,
	"window_type":"hamm",
	"n_mels": 32 #no. of mel bins
}

mlp_specs = {
    "input_type":"mel_spectrogram", #spectrogram or mel_spectrogram
	"input_window": 200, #duration of input window in ms [0,10,30,50,100,200]
	"p_train": 0.6, #percentage of data for training
	"p_val": 0.2,
	"p_test": 0.2,
	"path_to_save": path_to_save+"mlp/"
}

