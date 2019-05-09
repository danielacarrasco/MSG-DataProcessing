import numpy as np
import pandas as pd
import os
import gc
import pyedflib
import datetime
from scipy import signal


# Path to the file for processing, including the name of the .csv file.
loadPath = 'C:/Users/danie/Documents/Devices/Patients Data/Pt22_CF/'\
             'Output_Pt22CF.csv'

savePath = 'C:/Users/danie/Documents/Devices/Patients Data/Pt22_CF/'
device_n = '050716'

time_offset = 0  # Time offset in seconds.
t_offset = pd.DateOffset(seconds=time_offset)

def encodeIntEDF(data, channelInfo, sampleRate=100, dataType='int16'):

    channels = data.shape[1]    
    for c in range(channels):
        chanMin = channelInfo[c]['physical_min']
        chanMax = channelInfo[c]['physical_max']
        chanDiff = chanMax - chanMin
        digMin = channelInfo[c]['digital_min'] + 1
        digMax = channelInfo[c]['digital_max']
        digDiff = abs(digMin) + abs(digMax)
        with np.errstate(divide='ignore', invalid='ignore'):
            data[~np.isnan(data[:,c]),c] = (data[~np.isnan(data[:, c]), c] - 
                chanMin) / chanDiff * digDiff + digMin
            data[~np.isnan(data[:,c]),c] = 
                        np.clip(data[~np.isnan(data[:, c]), c], digMin, digMax)
    data = data.astype(np.dtype(dataType))
    data[np.isnan(data)] = digMin - 1
    data = data.reshape(-1, sampleRate, channels)
    data = np.transpose(data, (0, 2, 1))
    data = data.tostring(order='C')
    return data

def encodeEDF(data, sampleRate=100, dataType='float32'):

    channels = data.shape[1]
    data = data.astype(np.dtype(dataType))
    data = data.reshape(-1, sampleRate, channels)
    data = np.transpose(data, (0, 2, 1))
    data = data.tostring(order='C')
    return data

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

def makeEdf(fName, pat, startDateTime, df, device, cGroup, unit, 
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
    if 'f' in locals():
        f.close()
        del f
    
    fileName = fName + '/edf/' + studyTimeStr + '_' + cGroup + '.edf'
    
    f = pyedflib.EdfWriter(fileName, len(channelNames), 
                           file_type=pyedflib.FILETYPE_EDF)
    f.setStartdatetime(startDateTime)
    
    f.setGender('')
    f.setPatientName(pat)
    f.setSignalHeaders(channelInfo)
    print(newData)
    print(newData.shape)
    f.close()
    del f
    if edfType == 'int':
        newData = encodeIntEDF(newData, channelInfo, sampleRate=100)

    with open(fileName, 'r+') as f:
        f.seek(236)
        f.write(dataRecords)
    with open(fileName, 'rb+') as f:
        f.seek(0, 2)
        f.write(newData)

    

studyTimeStr = os.path.splitext(os.path.basename(loadPath))[0]


data = pd.read_csv(loadPath,skiprows=100,header=None,index_col=0)
data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S:%f") + t_offset


channelNames = ['Acc x','Acc y','Acc z','Light level','Button','T']

data = data.rename(columns={1:channelNames[0],2:channelNames[1],
                     3:channelNames[2],4:channelNames[3],
                     5:channelNames[4],6:channelNames[5]})

data = data.drop(columns='Button')

if not os.path.exists(savePath + '/edf'):
    os.makedirs(savePath + '/edf')

print(data.index[0])

studyDateTime =data.index[0]
offsetTime = datetime.timedelta()


data_acc =  pd.DataFrame(pd.concat([data['Acc x'], data['Acc y'], 
                                    data['Acc z']],axis=1))

data_acc['Acc Mag'] = np.sqrt(data['Acc x']*data['Acc x'] +
                      data['Acc y']*data['Acc y'] +
                      data['Acc z']*data['Acc z']) - 1

data_light = pd.DataFrame(pd.concat([data['Light level'].astype(float)],axis=1))

data_t = pd.DataFrame(pd.concat([data['T']],axis=1))


makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_light, device_n, 'Light Level', 'lux' , sampleRate=100, 
           units=None, edfType='int')

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_t, device_n, 'Temperature', 'deg C', sampleRate=100, 
           units=None, edfType='int')

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_acc, device_n, 'Acceleration', 'g', sampleRate=100, units=None, 
           edfType='int')
