import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes
from brainflow.data_filter import NoiseTypes, DetrendOperations, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes
# --- Configuration ---
BOARD_ID = BoardIds.CERELOG_X8_BOARD
SECONDS_TO_DISPLAY = 10
UPDATE_INTERVAL_MS = 40
Y_AXIS_PADDING_FACTOR = 1.2

# --- Global variables ---
board = None
eeg_channels = []
sampling_rate = 0
window_size = 0
data_buffer = np.array([])
y_limits = {}

def main():
    """
    Connects to the Cerelog board and creates a robust, real-time, scrolling plot
    with stable filtering and adaptive scaling.
    """
    global board, eeg_channels, sampling_rate, window_size, data_buffer, y_limits

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-1120"
    params.timeout = 15
    board = BoardShim(BOARD_ID, params)

    try:
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        window_size = SECONDS_TO_DISPLAY * sampling_rate

        if sampling_rate <= 0:
            raise BrainFlowError("Could not get a valid sampling rate from the board.", 0)

        for i in range(len(eeg_channels)):
            y_limits[i] = (-100, 100)

        print(f"Connecting to {board.get_board_descr(BOARD_ID)['name']}...")
        print(f"Detected Sampling Rate: {sampling_rate} Hz")
        board.prepare_session()
        print("\nStarting stream... Close the plot window to stop.")
        board.start_stream(5 * 60 * sampling_rate)
        time.sleep(2)

        num_board_channels = BoardShim.get_num_rows(BOARD_ID)
        data_buffer = np.empty((num_board_channels, 0))

        # --- Plot Setup ---
        fig, axes = plt.subplots(4, 2, figsize=(18, 10), sharex=True)
        fig.suptitle('Real-Time Cerelog EEG Waveforms (Correct Time Spacing)', fontsize=16)
        axes_flat = axes.flatten()
        lines = [ax.plot([], [], lw=1)[0] for ax in axes_flat]

        for i, ax in enumerate(axes_flat):
            ax.set_title(f'Channel {eeg_channels[i]}')
            ax.set_ylabel('Voltage (µV)')
            ax.grid(True)
            ax.set_xlim(-SECONDS_TO_DISPLAY, 0)

        fig.text(0.5, 0.04, 'Time (Seconds from "Now")', ha='center', va='center')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        def on_close(event):
            print("Plot window closed, stopping stream...")
            if board and board.is_prepared():
                board.stop_stream()
                board.release_session()
            print("Session released. Exiting.")

        fig.canvas.mpl_connect('close_event', on_close)

        ani = FuncAnimation(fig, update_plot, fargs=(lines, axes_flat), interval=UPDATE_INTERVAL_MS, blit=False)
        plt.show()

    except Exception as e:
        print(f"An error occurred in main(): {e}")
    finally:
        if board and board.is_prepared():
            board.release_session()

def update_plot(frame, lines, axes):
    """
    This function is called periodically to update the plot data.
    """
    global data_buffer, y_limits

    try:
        new_data = board.get_board_data()
        if new_data.shape[1] > 0:
            data_buffer = np.hstack((data_buffer, new_data))
            buffer_limit = int(window_size * 1.5)
            if data_buffer.shape[1] > buffer_limit:
                data_buffer = data_buffer[:, -buffer_limit:]

        plot_data = data_buffer[:, -window_size:]
        
        num_points = plot_data.shape[1]
        if num_points < 2:
            return

        eeg_plot_data = plot_data[eeg_channels] * 1e6
        
        # --- Filtering Logic (Corrected for Real-Time Stability) ---
        for i in range(len(eeg_channels)):
            # Use a safe data length check for the filters
            if eeg_plot_data[i].size > 20: 
                #1 Detrend to get dc offset away
                DataFilter.detrend(eeg_plot_data[i], DetrendOperations.CONSTANT.value)
                # 2. Apply a STABLE 4nd-order low-pass 100hz. This is crucial for real-time processing.
                DataFilter.perform_lowpass(eeg_plot_data[i], sampling_rate, 100.0, 4, FilterTypes.BUTTERWORTH, 0)
                
                # 3. Apply the band-stop (notch) filter for 50, 60 Hz noise.
                DataFilter.perform_bandstop(eeg_plot_data[i], sampling_rate, 48, 52, 3, FilterTypes.BUTTERWORTH, 0)
                DataFilter.perform_bandstop(eeg_plot_data[i], sampling_rate, 58, 62, 3, FilterTypes.BUTTERWORTH, 0)
                
                #4 High Pass above 0.5 Hz
                DataFilter.perform_highpass(eeg_plot_data[i], sampling_rate, 0.5, 4, FilterTypes.BUTTERWORTH, 0)
                
                #5. More cleaning data up
                #DataFilter.perform_rolling_filter(eeg_plot_data[i], 3, AggOperations.MEAN.value)
                ####DataFilter.perform_rolling_filter(eeg_plot_data[i], 3, AggOperations.MEDIAN.value)
                # (This is redundant notch) DataFilter.remove_environmental_noise(eeg_plot_data[i], sampling_rate, NoiseTypes.FIFTY_AND_SIXTY.value)
                #DataFilter.perform_wavelet_denoising(eeg_plot_data[i], WaveletTypes.BIOR3_9, 3,
                                                # WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                                # WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)
                
        # --- Manual Time Axis Generation (for True Scrolling) ---
        time_vector_full_window = np.linspace(-SECONDS_TO_DISPLAY, 0, window_size)
        time_vector_for_plot = time_vector_full_window[-num_points:]
        
        for i, (line, ax) in enumerate(zip(lines, axes)):
            channel_data = eeg_plot_data[i]
            
            # Check for invalid filter output (NaN) to prevent crashes
            if np.isnan(channel_data).any():
                print(f"Warning: NaN detected in channel {eeg_channels[i]} after filtering. Skipping one update.")
                continue
            
            centered_data = channel_data - np.mean(channel_data)
            
            line.set_data(time_vector_for_plot, centered_data)
            
            # --- Adaptive Y-Axis Logic ---
            # Define how many recent samples to use for auto-scaling (last 4 seconds)
            samples_for_scaling = int(4.0 * sampling_rate)
            recent_data = centered_data[-samples_for_scaling:]
            
            if recent_data.size > 0:
                max_val = np.max(recent_data)
                min_val = np.min(recent_data)
            else:
                max_val = np.max(centered_data)
                min_val = np.min(centered_data)
                
            if np.isclose(max_val, min_val):
                max_val += 1; min_val -= 1
                
            target_max = max_val * Y_AXIS_PADDING_FACTOR
            target_min = min_val * Y_AXIS_PADDING_FACTOR
            current_min, current_max = y_limits[i]
            smoothing_factor = 0.1
            new_max = current_max * (1 - smoothing_factor) + target_max * smoothing_factor
            new_min = current_min * (1 - smoothing_factor) + target_min * smoothing_factor
            ax.set_ylim(new_min, new_max)
            y_limits[i] = (new_min, new_max)

    except Exception as e:
        print(f"!!! ERROR IN UPDATE_PLOT: {e}")

if __name__ == "__main__":
    main()