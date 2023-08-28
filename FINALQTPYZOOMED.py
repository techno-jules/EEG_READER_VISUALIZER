#https://github.com/techno-jules/EEGML_Work/blob/main/qtpycompressionfinal
#Created by Julia Huang
import numpy as np
import eeghdf
import pyqtgraph as pg
from qtpy.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QLineEdit
import gzip
import eegvis.stacklineplot as stacklineplot
from scipy.signal import butter, lfilter
import eegml_signal.filters as esfilters
def load_EEG(eeg_name):
    DATADIR = "eegs" #use relative path

    EEGFILE = DATADIR+"/" + eeg_name
    
    eegf = eeghdf.Eeghdf(EEGFILE)
    #self.eegf = eegf
    #FIGSIZE = (12.0, 8.0)  
    #matplotlib.rcParams["figure.figsize"] = FIGSIZE
    return eegf

#high and low pass filter the EEG, neonotes 0.3 or 0.5 for high pass filtering, low pass 50/70, line up/share axes, no bar graphs
#add 4 graphs, 2 zoomed in by 10 secs each - pan by arrow buttpns, and 2 showing total range of graphs 
#show filtered and unfiltered eegs, calculate compression ratio after filtering
#vars that users can change
#4 graphs: original EEG, filtered EEG (both low and high pass), CR before filter, CR after both low and hogh pass filtering - (band pass filtering)
#https://colab.research.google.com/drive/1fOX1rSoJVyJ1n0QLv60OR7Eruljr4hkn
#filtered grpah most likely more sensitive to seizures
quantization_level=4 
sectotal=1000
num_sections=100
eeg_name = "eeg77.eeg.h5"
chan_number = 5 #channel #5 for EEG #show more channels afterwards, one channel shown at a time
eegf= load_EEG(eeg_name)
sample_rate = eegf.sample_frequency

def apply_lowpass_filter(signal, cutoff_freq, sampling_freq):
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(1, normalized_cutoff, btype='low')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def apply_highpass_filter(signal, cutoff_freq, sampling_freq):
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(1, normalized_cutoff, btype='high', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def compress_ratio(arr, quant=quantization_level):
           
    nbytes = arr.dtype.itemsize  
    nbits = 8 * nbytes
    rshift = nbits - quant
    shifted_arr = np.right_shift(arr, rshift)
    shifted_arr_bytes = bytes(shifted_arr.data)
    comp_arrbytes = gzip.compress(shifted_arr_bytes)

    return len(shifted_arr_bytes) / len(comp_arrbytes)

samp_freq=eegf.sample_frequency

hp_limit= 0.3#5 #1 #for actual model use 1 and 40 , but neonatal use 0.3 for hp
lp_limit = 50 #10 #40
lowpass = esfilters.fir_lowpass_firwin_ff(
            fs=samp_freq, cutoff_freq=lp_limit, numtaps=int(samp_freq / 4.0)
        )
 

highpass = esfilters.fir_highpass_firwin_ff(
            fs=samp_freq, cutoff_freq=hp_limit, numtaps=int(samp_freq)
        )
#lp_crthiswindow = lowpass(crthiswindow)
#hplp_crthiswindow = highpass(lp_crthiswindow)


def get_window(eeg):
    win_size_sec =1
    ii=30
    return eeg.rawsignals[:,int((ii*win_size_sec)*eeg.sample_frequency):int((ii+1)*win_size_sec*eeg.sample_frequency)]
  
thiswindow = get_window(eegf)
lowpassed = lowpass(thiswindow)
class SignalGrapher(QMainWindow):
    
    def __init__(self):
        super().__init__()
        #self.quantization_level=8
        central_widget = QWidget()
        self.setGeometry(100, 100, 3800, 3000)
        
        layout = QVBoxLayout(central_widget)
        self.label = QLabel("Welcome to the EEG Signal Grapher QTPY App! Made by Julia Huang")
        self.label.setStyleSheet("font-size: 12pt; font-weight: bold; color: purple; margin: 1px; background-color: pink;")

        self.label2 = QLabel("Current User Settings: | EEG Name: "+eeg_name+" | EEG Channel # Shown: "+str(chan_number)+" | Quantization Level: "+str(quantization_level)+" | # Secs of EEG to Display: "+str(sectotal)+" | # Sections to Calculate CR: "+str(num_sections))
        self.label2.setStyleSheet("font-size: 10pt; font-weight: bold; color: blue; background-color: pink;")
        #self.label3 = QLabel("Quantization Level: "+str(quantization_level))
    

        self.button1 = QPushButton("Display Numerical Compression Ratios")
        self.button1.setStyleSheet("font-size: 10pt; font-weight: bold; color: blue; margin: 1px; background-color: #F1BDFF;")
        self.button1.clicked.connect(self.show_message)
        self.lineedit1 = QLineEdit()
        self.lineedit2 = QLineEdit()
        
        self.plot_widget1 = pg.PlotWidget()
        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget3 = pg.PlotWidget()
        self.plot_widget4 = pg.PlotWidget()
        self.plot_widget5 = pg.PlotWidget()
        self.plot_widget6 = pg.PlotWidget()
        self.plot_widget7 = pg.PlotWidget()
        self.plot_widget8 = pg.PlotWidget() #filtered eeg graphs
        self.plot_widget9 = pg.PlotWidget() #filtered eeg graphs
        self.plot_widgetbar = pg.PlotWidget()

        layout.addWidget(self.label)
        layout.addWidget(self.label2)
        #sin waves
        #layout.addWidget(self.plot_widget1)
        #layout.addWidget(self.plot_widget2)
        #layout.addWidget(self.plot_widget3)
        #eeg signals
        layout.addWidget(self.plot_widget5) #full zoomed out eeg graph
        layout.addWidget(self.plot_widget4) #compression ratio of full zoomed out eeg graph
        #layout.addWidget(self.plot_widget8) #filtered eeg grpahs
        #layout.addWidget(self.plot_widget9) #filtered eeg grpahs
        layout.addWidget(self.button1)
        layout.addWidget(self.plot_widget7) #compression ratio of zoomed in eeg graph
        layout.addWidget(self.plot_widget6) #zoomed in eeg graph

       

        self.left_button = QPushButton("Left 10 Seconds")
        self.left_button.clicked.connect(self.move_left)
        layout.addWidget(self.left_button)

        

        self.right_button = QPushButton("Right 10 Seconds")
        self.right_button.clicked.connect(self.move_right)
        layout.addWidget(self.right_button)

        self.left_button.setStyleSheet("font-size: 10pt; font-weight: bold; color: blue; margin: 1px; background-color: #F1BDFF;")
        self.right_button.setStyleSheet("font-size: 10pt; font-weight: bold; color: blue; margin: 1px; background-color: #F1BDFF;")

        self.current_x_range = (0, sectotal/50) #custom x-range for zoomed in graphs
        
        
        #layout.addWidget(self.plot_widgetbar)

        # Set the central widget
        self.setCentralWidget(central_widget)

        # Generate the custom signals
        self.generate_signals()

        # Plot the signals
        self.plot_signals()

    def move_left(self):
        tick_range = 10
        self.current_x_range = (self.current_x_range[0] - tick_range, self.current_x_range[1] - tick_range)
        #self.plot_widget7.clear()
        #self.plot_widget6.clear()
        self.plot_signals()

    def move_right(self):
        tick_range = 10
        self.current_x_range = (self.current_x_range[0] + tick_range, self.current_x_range[1] + tick_range)
        #self.plot_widget7.clear()
        #self.plot_widget6.clear()
        self.plot_signals()

    def show_message(self):
        
        message = "<p style='font-size: 20px; color: purple; font-weight: 800; text-align: center; background-color: #ECA6FF;'>Compression Ratio Calculations \n "
        message+="<p style='font-size: 16px; color: blue;'>CR of Each EEG Section: \n </p>" 
        #message+="\n " + self.CRs + "\n \n" 
        message+="\n " + str(self.CR) + "\n \n"
        message+="<p style='font-size: 16px; color: blue;'>Average Combined CR of entire EEG file: </p>" 
        message += str(self.CRsAvg) + "\n "
        QMessageBox.information(self, "Compression Ratio Calculations", message)
        #print(self.CRs)

    def generate_signals(self):
        fs = 1000  # Sampling frequency (Hz)
        f_signal = 10  # Signal frequency (Hz)
        f_signal2 = 20
        f_signal3 = 30
        self.sinx = np.arange(0, 1, 1/fs)  # Time vector
        self.signal=np.sin(2*np.pi*f_signal*self.sinx)+np.sin(2*np.pi*f_signal2*self.sinx)+np.sin(2*np.pi*f_signal3*self.sinx)
        
        # Generate low-frequency noise
        f_noise_low = 2  # Low-frequency noise frequency (Hz)
        noise_low = np.sin(2 * np.pi * f_noise_low * self.sinx)

        # Generate high-frequency noise
        f_noise_high = 200  # High-frequency noise frequency (Hz)
        noise_high = np.sin(2 * np.pi * f_noise_high * self.sinx) #change frequency of sine wave

        # Add noise to the original signal
        x_with_noise = self.sinx  + 0.1 * noise_low + 0.05 * noise_high #adding additive noise
       
        # Apply filters to the noisy signal
        cutoff_freq_lowpass = 30  # Low-pass filter cutoff frequency (Hz)
        cutoff_freq_highpass = 80  # High-pass filter cutoff frequency (Hz)
        x_lowpass = apply_lowpass_filter(x_with_noise, cutoff_freq_lowpass, fs)
        x_highpass = apply_highpass_filter(x_with_noise, cutoff_freq_highpass, fs)
        self.signal2=x_lowpass
        self.signal3=x_highpass
        eegf=load_EEG(eeg_name)
        #sample_rate = getSampleRate(eeg_name)
        '''
        DATADIR = "eegs" #use relative path

        EEGFILE = DATADIR + "/absence_epilepsy.eeg.h5"
        
        eegf = eeghdf.Eeghdf(EEGFILE)
        self.eegf = eegf
        FIGSIZE = (12.0, 8.0)  
        matplotlib.rcParams["figure.figsize"] = FIGSIZE
        chan_number = 4
        sample_rate = eegf.sample_frequency
        '''
        
        '''
        self.sec=25
        self.sec2=50
        self.sec3=75
        '''
        #vars that users can change
        #self.sectotal=100
        #self.num_sections=10

        #--
        self.num_secs_list = []
        self.bysecs=int(sectotal/num_sections)
        
        for x in range(self.bysecs, sectotal, self.bysecs):
            self.num_secs_list.append(x)
            x+=self.bysecs
        
        self.eeg_sections = []
        eeg_sections_string = []
        for x in range(num_sections):
            self.eeg_sections.append(x)
            eeg_sections_string.append("eeg"+str(x))
            

        last_section = self.eeg_sections[(len(self.eeg_sections)-1)]
        secondlast_section = self.eeg_sections[(len(self.eeg_sections)-2)]
        last_sec = self.num_secs_list[(len(self.num_secs_list)-1)]
        #print(self.num_secs_list)
        self.eeg_list = []
        for x in eeg_sections_string:
            if x == "eeg0":
                #x = "0 to "+str(self.num_secs_list[0])
                #print(x) 
                self.firstsignal = eegf.rawsignals[chan_number, 0:int(sample_rate*self.num_secs_list[0])] 
           
        self.eeg_list.append(self.firstsignal)

        for x in self.num_secs_list:
            #print(str(x)+" to "+str(x+bysecs))
            self.eeg_list.append(eegf.rawsignals[chan_number, int(sample_rate * x):int(sample_rate*(x+self.bysecs))])
        '''
        f10_0 = eegf.rawsignals[chan_number, 0:int(sample_rate*self.sec)] #0 to 25
        f10_1 = eegf.rawsignals[chan_number, int(sample_rate * self.sec):int(sample_rate*self.sec2)] #25 to 50
        f10_2 = eegf.rawsignals[chan_number, int(sample_rate * self.sec2):int(sample_rate*self.sec3)] #50 to 75
        f10_3 = eegf.rawsignals[chan_number, int(sample_rate * self.sec3):int(sample_rate*self.sectotal)] #75 to 100
        '''
        first_numsecs = eegf.rawsignals[chan_number, 0:int(sample_rate*sectotal)]
        
        self.compression_samplerate=0.1
        self.seg_compare = []
        
        for x in self.eeg_sections:
            self.seg_compare.append((compress_ratio(self.eeg_list[x], quant=quantization_level)))

        self.seg_compare=np.array(self.seg_compare)
        #seg_compare = np.array([compress_ratio(f10_0, quant=8), compress_ratio(f10_1, quant=8),compress_ratio(f10_2, quant=8),compress_ratio(f10_3, quant=8)])
        self.t2 = np.arange(self.seg_compare.shape[0])/self.compression_samplerate #+ 0.5 *(1/self.compression_samplerate)# add an offset of 0.5 *(1/self.compression_samplerate), to put in center of graph
        self.t = np.arange(first_numsecs.shape[0])/sample_rate 
        '''
        self.firstsignal=f10_0
        self.secsignal=f10_1
        self.thirdsignal=f10_2
        self.fourthsignal=f10_3
        '''
        #print(len(self.seg_compare))
        #print(self.seg_compare)
        
        self.signal4= self.seg_compare
        self.signal5= first_numsecs
       
        
    def plot_signals(self):
        # Clear the previous plot
        self.plot_widget1.clear()
        self.plot_widget2.clear()
        self.plot_widget3.clear()
        self.plot_widget4.clear()
        self.plot_widget5.clear()
        self.plot_widget6.clear()
        self.plot_widget7.clear()
        self.plot_widget8.clear()
        self.plot_widget9.clear()
        self.plot_widgetbar.clear()

        # Plot the custom signals
        self.plot_widget1.plot(x=self.sinx, y=self.signal, pen='c')
        self.plot_widget1.disableAutoRange()
        self.plot_widget1.setTitle("3 Sin Waves Signal")
        self.plot_widget1.setLabel('left', 'Amplitude')
        self.plot_widget1.setLabel('bottom', 'Time')

        self.plot_widget2.plot(x=self.sinx, y=self.signal2, pen='r')
        self.plot_widget2.disableAutoRange()
        self.plot_widget2.setTitle("Sin Wave Signal Low Pass Filter")
        self.plot_widget2.setLabel('left', 'Amplitude')
        self.plot_widget2.setLabel('bottom', 'Time')

        self.plot_widget3.plot(x=self.sinx, y=self.signal3, pen='y')
        self.plot_widget3.disableAutoRange()
        self.plot_widget3.setTitle("Sin Wave High Pass Filter")
        self.plot_widget3.setLabel('left', 'Amplitude')
        self.plot_widget3.setLabel('bottom', 'Time')

        # Create the bar graph item
        '''
        self.bar_graph_item = pg.BarGraphItem(x=np.arange(4), height=[np.array(compress_ratio(self.firstsignal, quant=8)), np.array(compress_ratio(self.secsignal, quant=8)),np.array(compress_ratio(self.thirdsignal, quant=8)),np.array(compress_ratio(self.fourthsignal, quant=8)), ], width=0.25, brush='g')

        '''
        seg_compare_bars = []
        
        for x in self.eeg_sections:
            seg_compare_bars.append(np.array((compress_ratio(self.eeg_list[x], quant=quantization_level))))

        self.seg_compare=np.array(self.seg_compare)
        self.bar_graph_item = pg.BarGraphItem(x=np.arange(num_sections), height=[seg_compare_bars], width=0.25, brush='g')

        # Create a plot widget and add the bar graph item to it
        self.plot_widgetbar.addItem(self.bar_graph_item)
        self.plot_widgetbar.setTitle("EEG Compression Ratios as Bar Graphs: "+ str(num_sections)+ " sections"+" (calculate CR every "+str(self.bysecs)+" secs)")

        # Set the labels for x and y axes
        self.plot_widgetbar.setLabel('left', 'Compression Ratio (CR)')
        self.plot_widgetbar.setLabel('bottom', 'segment')

        num_bars = self.eeg_sections
        bar_labels = []
        #tuple1 = ("apple", "banana", "cherry")
        bar_labels.append(str(0)+"-"+ str(self.bysecs)+" s")
        for x in self.num_secs_list: 
            bar_labels.append(str(x)+"-"+ str(x+self.bysecs)+" s")

        bars = list(zip(num_bars, bar_labels))

        # Set the ticks for x-axis
        x_axis = pg.AxisItem(orientation='bottom')
        #x_axis.setTicks([[(0, '0-25 seconds CR'), (1, '25-50 seconds CR'), (2, '50-75 seconds CR'), (3, '75-100 seconds CR')]])
        x_axis.setTicks([bars])

        # Add the x-axis to the plot widget
        self.plot_widgetbar.getPlotItem().setAxisItems({'bottom': x_axis})


        eegf=load_EEG(eeg_name)

        self.plot_widget4.plot(x=self.t2, y=self.signal4, pen='w')
        #self.plot_widget4.disableAutoRange()
        self.plot_widget4.setYRange(0.0, 100.0)
        self.plot_widget4.setXRange(0, sectotal, padding=0.11)
        self.plot_widget4.setTitle((f"Compression Ratio of {num_sections} sections in secs: {eeg_name}"))
        self.plot_widget4.setLabel('left', 'compression ratio')
        self.plot_widget4.setLabel('bottom', 'seconds')

        self.plot_widget5.plot(x=self.t, y=self.signal5, pen='m')
        #self.plot_widget5.setXRange(0.0, sectotal-5, padding=0.09)
        self.plot_widget5.setTitle((f"1st {sectotal} secs of EEG signal: {eeg_name}"))
        self.plot_widget5.setLabel('left', 'amplitude')
        self.plot_widget5.setLabel('bottom', 'seconds')

        self.plot_widget5.setXLink(self.plot_widget4)

        #zoomed in graphs below ################

        self.plot_widget6.plot(x=self.t2, y=self.signal4, pen='w')
        #self.plot_widget4.disableAutoRange()
        self.plot_widget6.setYRange(0.0, 100.0)
        self.plot_widget6.setXRange(self.current_x_range[0], self.current_x_range[1], padding=0.11)
        self.plot_widget6.setTitle((f"ZOOMED IN: Compression Ratio of {num_sections} sections in secs: {eeg_name}"))
        self.plot_widget6.setLabel('left', 'compression ratio')
        self.plot_widget6.setLabel('bottom', 'seconds')
        #self.current_x_range[0], self.current_x_range[1]

        self.plot_widget7.plot(x=self.t, y=self.signal5, pen='m')
        #self.plot_widget5.setXRange(0.0, sectotal-5, padding=0.09)
        self.plot_widget7.setTitle((f"ZOOMED IN: 1st {sectotal} secs of EEG signal: {eeg_name}"))
        self.plot_widget7.setLabel('left', 'amplitude')
        self.plot_widget7.setLabel('bottom', 'seconds')

        self.plot_widget7.setXLink(self.plot_widget6)

        
        ####filtered eeg graphs
        self.plot_widget8.plot(x=self.t, y=lowpass(self.signal5), pen='w')
        #self.plot_widget5.setXRange(0.0, sectotal-5, padding=0.09)
        self.plot_widget8.setTitle(("EEG Through Low-Pass Filter"))
        self.plot_widget8.setLabel('left', 'amplitude')
        self.plot_widget8.setLabel('bottom', 'seconds')
         ####filtered eeg graphs
        self.plot_widget9.plot(x=self.t, y=highpass(self.signal5), pen='y')
        #self.plot_widget5.setXRange(0.0, sectotal-5, padding=0.09)
        self.plot_widget9.setTitle(("EEG Through High-Pass Filter"))
        self.plot_widget9.setLabel('left', 'amplitude')
        self.plot_widget9.setLabel('bottom', 'seconds')

        

        # Refresh the plots
        #self.plot_widget1.autoRange()
        self.plot_widget2.autoRange()
        self.plot_widget3.autoRange()
        #self.plot_widget4.autoRange()
        self.plot_widget5.autoRange()
        self.plot_widgetbar.autoRange()
        '''
        cr1 = np.array(compress_ratio(self.firstsignal, quant=8))
        cr2 = (np.array(compress_ratio(self.secsignal, quant=8)))
        cr3 = (np.array(compress_ratio(self.thirdsignal, quant=8)))
        cr4 = (np.array(compress_ratio(self.fourthsignal, quant=8)))
        '''
        '''
        CR = [cr1, cr2, cr3, cr4]
        CRfinal = [str(cr1), str(cr2), str(cr3), str(cr4)]
        self.CRs = " "
        self.CRs = self.CRs.join(CRfinal)
        self.CRsAvg = sum(CR) / len(CR)
        '''
        '''
        CR = self.seg_compare
        CRfinal = str(CR)
        self.CRs = " "
        self.CRS = self.CRs.join(CRfinal)
        #print(self.CRs)
        self.CRsAvg = sum(CR) / len(CR)
        '''
        self.CR = self.seg_compare
        #self.CRfinal = []
        #for x in CR:
            #self.CRfinal.append(str(x))
        self.CRsAvg = sum(self.CR) / len(self.CR)
            


        
# Create the application
app = QApplication([])

# Create a QtPy application window
window = SignalGrapher()
window.setWindowTitle("EEG Signal Grapher Final by Julia Huang")
window.resize(800, 600)

# Show the window
window.show()

# Start the application event loop
app.exec_()
