import numpy as np
import pandas as pd
import os
import pyedflib


# Path to the file for processing, including the name of the .csv file.
loadPath = 'C:/Users/danie/Documents/Devices/Patients Data/Pt35_KDP/'\
           'Output_Pt35KDP.csv'

savePath = 'C:/Users/danie/Documents/Devices/Patients Data/Pt35_KDP/'

time_offset = 0  # Time offset in seconds.
t_offset = pd.DateOffset(seconds=time_offset)

sampleRate = 100.0 # Sample rate including correction

print("Patient "+ savePath.split('/')[-2])

def makeChannelHeaders(label, unit=None, sampleRate=100.0,
                        physicalMax=10, physicalMin=-10,
                        digital_max=32767, digital_min=-32768,
                        transducer='', prefilter=''):
    
    ch_dict = {'label': label, 'dimension': unit, 
       'sample_rate': sampleRate, 'physical_max': physicalMax, 
       'physical_min': physicalMin, 'digital_max': digital_max, 
       'digital_min': digital_min, 'transducer': transducer, 
       'prefilter':prefilter}
    
    return ch_dict

def makeEdf(fName, pat, startDateTime, df, cGroup, unit, 
            sampleRate=100.0, units=None, edfType='int'):
    channelInfo = []
    dataList = []
    channelNames = df.columns.values.tolist()
    dataRecords = str(int(np.ceil(len(df)/sampleRate)))
    newData = np.full(df.shape, np.nan, dtype=np.float32)
    if units is None:
        units = ['V' for i in range(df.shape[1])]
    
    for c in range(len(channelNames)):
        cName = str(channelNames[c])
        print(cName)
        transducer = cGroup
        x = np.asarray(df.iloc[:,c])
        exponent = 1
        newData[:,c] = np.asarray(df.iloc[:,c].copy()) * exponent
        
        if edfType == 'int':
            lowerBound = np.min(df.min())
            upperBound = np.max(df.max())

        else:
            lowerBound = 0
            upperBound = 0
        
        ch_dict = makeChannelHeaders(cName,
                                     unit=unit,
                                     sampleRate=sampleRate,
                                     physicalMax=upperBound,
                                     physicalMin=lowerBound,
                                     transducer=transducer)
        
        channelInfo.append(ch_dict)
        dataList.append(x)
    if 'f' in locals():
        f.close()
        del f
    
    fileName = fName + '/edf_upload/' + studyTimeStr + '_' + cGroup + '.edf'
    
    f = pyedflib.EdfWriter(fileName, len(channelNames), 
                           file_type=pyedflib.FILETYPE_EDF)
    f.setStartdatetime(startDateTime)
    
    f.setGender('')
    f.setPatientName(pat)
    f.setSignalHeaders(channelInfo)
    new_rate = 100 # Dummy rate to enter the loop
    c = 0.1
    while new_rate > 60:
        c = c * 10
        new_rate = (1000/c)*1./sampleRate
    for i in range(len(channelNames)):
        f.setSamplefrequency(i,1000/c)
    f.setDatarecordDuration(new_rate*100000)
    f.writeSamples(dataList)
    f.close()
    del f

    
studyTimeStr = os.path.splitext(os.path.basename(loadPath))[0]


data = pd.read_csv(loadPath, skiprows=100, header=None, index_col=0)
data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S:%f") + t_offset


channelNames = ['Acc x','Acc y','Acc z','Light level','Button','T']

data = data.rename(columns={1:channelNames[0],2:channelNames[1],
                     3:channelNames[2],4:channelNames[3],
                     5:channelNames[4],6:channelNames[5]})

data = data.drop(columns='Button')

if not os.path.exists(savePath + '/edf_upload'):
    os.makedirs(savePath + '/edf_upload')

print(data.index[0])

studyDateTime = data.index[0] + t_offset


data_acc =  pd.DataFrame(pd.concat([data['Acc x'], data['Acc y'], 
                                    data['Acc z']],axis=1))

data_acc['Acc Mag'] = np.sqrt(data['Acc x']*data['Acc x'] +
                      data['Acc y']*data['Acc y'] +
                      data['Acc z']*data['Acc z']) - 1

data_light = pd.DataFrame(pd.concat([data['Light level'].astype(float)],axis=1))

data_t = pd.DataFrame(pd.concat([data['T']],axis=1))

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_light, 'Light Level', 'lux' , sampleRate, 
           units=None, edfType='int')

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_t, 'Temperature', 'deg C', sampleRate, 
           units=None, edfType='int')

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_acc, 'Acceleration', 'g', sampleRate, units=None, 
           edfType='int')


