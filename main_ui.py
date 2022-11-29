import recorder
import speaker as sp
import PySimpleGUI as sg
from threading import Thread
from recognize.predict import predict
import recognize.train as tn
import recognize.main as mn
from config import TEMP_DIR
import threading
import os


# Pages
recorder = recorder.Recorder(recorder.StreamParams())
speaker = sp.Speaker()
_main_btns_block = False

textAI = sg.Text('...', font='Helvetica 20')
_label = False

def main():
    
    yesBtn = sg.ReadButton('Yes', key = "Yes", button_color='green')
    noBtn = sg.ReadButton('No', key = "No", button_color='darkred')
    yesnoColumn = sg.Column([[yesBtn, noBtn]], visible=False,  k='-C-')
    
    def get_predict_output():
        output = ""
        
        global _label
        _label, _confidence = predict(os.path.join(TEMP_DIR, recorder._last_audio_name))
        if _label != None:
            if _label != False:
                yesnoColumn.update(visible=True)
                output = "Are you "+ _label +"? "+ _confidence
            else:
                output = "Can you repeat that please?"
        else:
            output = "Train me before using."
            
        textAI.update(value = output)
        
    # talkBox = sg.Listbox(values=["..."], size=(30, 6))
    recordBtn = sg.ReadButton('Record', key = "Record")
    stopBtn = sg.ReadButton('Stop', key = "Stop", disabled=True)
    resetBtn = sg.ReadButton('Reset everything', key = "Reset")
    timeTracker = sg.Text('', visible= False)
    
    layout = [[sg.Text('Recognize', font='Helvetica 15'), timeTracker],
            [recordBtn, stopBtn, sg.ReadButton('Training Queue', key = "TrainQueue"), sg.ReadButton('Console', key = "Console", disabled=True), sg.ReadButton('Dataset', key = "Dataset")],
            [sg.Column([[textAI]], vertical_alignment='center', justification='center',  k='-C-', pad=(0, (20, 5)))],
            [sg.Column([[yesnoColumn]],  k='-C-', pad=(0, (10, 20)), vertical_alignment='center', justification='center')],
            [sg.Column([[resetBtn, sg.Exit()]],  k='-C-', justification='right')]]

    window = sg.Window('Speech Recognition', layout)
    global _main_btns_block

    while True:
        event, values = window.Read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            remove_temp_audio()
            break
        
        elif event == 'Record':
         #   threading.Thread(target=write, args=("hello",), daemon=True).start()
            recorder.start_recording(timeTracker)
            recordBtn.update(disabled = True)
            stopBtn.update(disabled = False)
            resetBtn.update(disabled = True)
            yesnoColumn.update(visible=False)
            textAI.update(value = "...")
            timeTracker.update(visible=True)
            
        elif event == 'Stop':
            if recorder._is_started:
                recordBtn.update(disabled = False)
                stopBtn.update(disabled = True)
                resetBtn.update(disabled = False)
                recorder.stop_recording()
                new_thread = Thread(target=get_predict_output, args=(), daemon=True)
                new_thread.start()
                
        elif event == 'Reset':
            result = sg.PopupYesNo("Are you sure about this? You really want to reset everything to default??")
            if result == 'Yes':
                resetBtn.update(disabled = True)
                recordBtn.update(disabled = True)
                _main_btns_block = True
                # console_window()
                mn.reset_everything()
                    
        elif event == 'Dataset':
            dataset_window()
            
        elif event == 'TrainQueue':
            train_queue_window()
       
        # elif event == 'Console':
            # console_window()
            
            
        elif event == 'Yes':
            yesnoColumn.update(visible=False)
            textAI.update(value="My training was not in vain...")
            timeTracker.update(visible=False)
            mn.add_record(_label, recorder._last_audio_name)
            # TODO: learn more
            remove_temp_audio()
            
        elif event == 'No':
            yesnoColumn.update(visible=False)
            textAI.update(value="")
            timeTracker.update(visible=False)
            
            
            who_are_you_window()
            remove_temp_audio()
            
        if _main_btns_block and tn.is_running == False:
            resetBtn.update(disabled = False)
            recordBtn.update(disabled = False)
            _main_btns_block = False
                    
    window.close()
    

def who_are_you_window():
    layout = [[sg.Text("Who are you then?")],
              [sg.Input(do_not_clear=True, size=(20,1),enable_events=True, key='_INPUT_')],
            [sg.Column([[sg.ReadButton('Submit', key = "Submit")]], vertical_alignment='center', justification='center',  k='-C-', pad=(0, 10))]]

    window = sg.Window('...?', layout)

    while True:
        event, values = window.Read()
        if event == sg.WIN_CLOSED:
            textAI.update(value="Whatever...")
            remove_temp_audio()
            break
        elif event == "Submit":  
            _classe_name = values['_INPUT_']
            if _classe_name != '':
                mn.add_record(_classe_name, recorder._last_audio_name)
                textAI.update(value="Hmm...")       
            else:
                textAI.update(value="Whatever...")
            break
        
    window.close()
    
    
def train_queue_window():
    def getAllInQueue():
        queue = []
        for item in tn.get_threads_in_line():
            queue.append(item["why"])
            
        return queue
    
    layout = [[sg.Listbox(values=getAllInQueue(), size=(30, 6))],
            [sg.Exit()]]

    window = sg.Window('Training Queue', layout)

    while True:
        event, values = window.Read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()

        
def dataset_window():
    deleteBtn = sg.ReadButton('Delete', key = "Delete", disabled=True)
    
    layout = [[sg.Listbox(values=mn.get_dataset_fullness(), size=(30, 6), enable_events=True, key='_LIST_')],
            [deleteBtn, sg.Exit()]]

    window = sg.Window('Dataset', layout)
    global _main_btns_block
    
    while True:
        event, values = window.Read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        elif event == '_LIST_' and len(values['_LIST_']):     # if a list item is chosen
           # sg.Popup('Selected ', values['_LIST_'])
            if _main_btns_block == False and deleteBtn.Disabled == True:
                deleteBtn.update(disabled=False)
                
        elif event == "Delete":
            selectedValue = values['_LIST_'][0].split()[0]
            if mn.remove_person(selectedValue):
                _main_btns_block = True
                deleteBtn.update(disabled=True) 
            else:
                sg.Popup('Sorry, CANNOT DELETE! At least one full record must be left for the model!')
                
           
    window.close()

    
# def console_window():
#     init = True
    
#     layout = [[ sg.Output(size=(80, 10), visible=False)],
#             [sg.Exit()]]

#     window = sg.Window('Console', layout, size=(100, 100))
        
#     while True:
#         event, values = window.Read()
#         if event == "Exit" or event == sg.WIN_CLOSED:
#             break
    
        
#     window.close()
        
    
    

def remove_temp_audio():
    # remove temp audio file
    if ( recorder._last_audio_name != None):
        temp_wav_path = os.path.join(TEMP_DIR, recorder._last_audio_name)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
    
if __name__ == '__main__':
    main()