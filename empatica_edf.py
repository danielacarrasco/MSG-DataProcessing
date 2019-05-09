import numpy as np
import pandas as pd
import os
import pyedflib
from scipy import signal


# Path to the files for processing.
loadPath = 'C:/Users/danie/Documents/Devices/Patients Data/Pt26_MO/Session2/'
# Path where the folder (edf/) with the EDF files will be saved. 
savePath = 'C:/Users/danie/Documents/Devices/Patients Data/Pt26_MO/Session2/'

time_offset = 0  # Time offset in seconds.
t_offset = pd.DateOffset(seconds=time_offset)

def encodeIntEDF(data, channelInfo, sampleRate=75, dataType='int16'):

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
                          np.clip(data[~np.isnan(data[:,c]),c], digMin, digMax)
    data = data.astype(np.dtype(dataType))
    data[np.isnan(data)] = digMin - 1
    data = data.reshape(-1, sampleRate, channels)
    data = np.transpose(data, (0, 2, 1))
    data = data.tostring(order='C')
    return data

def encodeEDF(data, sampleRate=75, dataType='float32'):

    channels = data.shape[1]
    data = data.astype(np.dtype(dataType))
    data = data.reshape(-1, sampleRate, channels)
    data = np.transpose(data, (0, 2, 1))
    data = data.tostring(order='C')
    return data

def makeChannelHeaders(label, unit=None, sampleRate=75.0,
                        physicalMax=10, physicalMin=-10,
                        digital_max=32767, digital_min=-32768,
                        transducer='', prefilter=''):
    
    ch_dict = {'label': label, 'dimension': unit, 
       'sample_rate': sampleRate, 'physical_max': physicalMax, 
       'physical_min': physicalMin, 'digital_max': digital_max, 
       'digital_min': digital_min, 'transducer': transducer, 
       'prefilter':prefilter}
    
    return ch_dict

def makeEdf(fName, pat, startDateTime, df, cGroup, unit, sampleRate, 
            units=None, edfType='int'):
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
        #unit = 'V'
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
    #print(newData)
    print(newData.shape)
    f.close()
    del f
    if edfType == 'int':
        newData = encodeIntEDF(newData, channelInfo, sampleRate)


    with open(fileName, 'r+') as f:
        f.seek(236)
        f.write(dataRecords)
    with open(fileName, 'rb+') as f:
        f.seek(0, 2)
        f.write(newData)


studyTimeStr = os.path.splitext(os.path.basename(loadPath))[0]

channelNames = ['Acc x','Acc y','Acc z','BVP','EDA','HR','IBI','T']

data_acc = pd.read_csv(loadPath+'ACC.csv',header=None)
print(data_acc[0][0])
t0 = pd.to_datetime(data_acc[0][0], unit="s")
f = 1./data_acc[0][1]
#print(f)
t_index = pd.Series(pd.date_range(t0,periods=len(data_acc)-2, freq=str(f)+'S'))
t_index = t_index.dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
t_index = pd.to_datetime(t_index)
t_index = t_index.dt.tz_localize(None)
data_ac = pd.DataFrame(pd.concat([data_acc[0][2:], data_acc[1][2:],
                                  data_acc[2][2:]],axis=1))
data_ac.index = t_index + t_offset
data_ac.rename(columns={0:"Acc x", 1:"Acc y", 2:"Acc z"}, inplace=True)
data_ac['Acc Mag'] = np.sqrt(data_ac['Acc x']*data_ac['Acc x'] +
                      data_ac['Acc y']*data_ac['Acc y'] +
                      data_ac['Acc z']*data_ac['Acc z']) - 1
      
data_bvp = pd.read_csv(loadPath+'BVP.csv',header=None)
t0 = pd.to_datetime(data_bvp[0][0], unit="s")
f = 1./data_bvp[0][1]
t_index = pd.Series(pd.date_range(t0,periods=len(data_bvp)-2, freq=str(f)+'S'))
t_index = t_index.dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
t_index = pd.to_datetime(t_index)
t_index = t_index.dt.tz_localize(None)
data_b = pd.DataFrame(pd.concat([data_bvp[0][2:]],axis=1))
data_b.index = t_index + t_offset
data_b.rename(columns={0:"BVP"}, inplace=True)

data_eda = pd.read_csv(loadPath+'EDA.csv',header=None)
t0 = pd.to_datetime(data_eda[0][0], unit="s")
f = 1./data_eda[0][1]
t_index = pd.Series(pd.date_range(t0,periods=len(data_eda)-2, freq=str(f)+'S'))
t_index = t_index.dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
t_index = pd.to_datetime(t_index)
t_index = t_index.dt.tz_localize(None)
data_e = pd.DataFrame(pd.concat([data_eda[0][2:]],axis=1))
data_e.index = t_index + t_offset
data_e.rename(columns={0:"EDA"}, inplace=True)

data_hr = pd.read_csv(loadPath+'HR.csv',header=None)
t0 = pd.to_datetime(data_hr[0][0], unit="s")
f = 1./data_hr[0][1]
t_index = pd.Series(pd.date_range(t0,periods=len(data_hr)-2, freq=str(f)+'S'))
t_index = t_index.dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
t_index = pd.to_datetime(t_index)
t_index = t_index.dt.tz_localize(None)
data_hrate =  pd.DataFrame(pd.concat([data_hr[0][2:]],axis=1))
data_hrate.index = t_index + t_offset
data_hrate.rename(columns={0:"HR"}, inplace=True)

data_temp = pd.read_csv(loadPath+'TEMP.csv',header=None)
t0 = pd.to_datetime(data_temp[0][0], unit="s")
f = 1./data_temp[0][1]
t_index = pd.Series(pd.date_range(t0,periods=len(data_temp)-2, 
                                  freq=str(f)+'S'))
t_index = t_index.dt.tz_localize('UTC').dt.tz_convert('Australia/Melbourne')
t_index = pd.to_datetime(t_index)
t_index = t_index.dt.tz_localize(None)
data_t = pd.DataFrame(pd.concat([data_temp[0][2:]],axis=1))
data_t.index = t_index + t_offset
data_t.rename(columns={0:"T"}, inplace=True)


if not os.path.exists(savePath + '/edf'):
    os.makedirs(savePath + '/edf')

studyDateTime = data_ac.index[0]

while(data_ac.shape[0] % int(data_acc[0][1]) != 0):
    cuoc = data_ac.shape[0] % int(data_acc[0][1])
    data_ac.drop(data_ac.tail(cuoc).index,inplace=True)

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_ac, 'Acc', 'g' , sampleRate=int(data_acc[0][1]), units=None, 
           edfType='int')


studyDateTime = data_b.index[0]

while(data_b.shape[0] % int(data_bvp[0][1]) != 0):
    cuoc = data_b.shape[0] % int(data_bvp[0][1])
    data_b.drop(data_b.tail(cuoc).index,inplace=True)
    
makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_b, 'BVP', 'XXXX', sampleRate=int(data_bvp[0][1]), units=None, 
           edfType='int')


studyDateTime = data_e.index[0]

while(data_e.shape[0] % int(data_eda[0][1]) != 0):
    cuoc = data_e.shape[0] % int(data_eda[0][1])
    data_e.drop(data_e.tail(cuoc).index,inplace=True)

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_e, 'EDA', 'XXXX', sampleRate=int(data_eda[0][1]), units=None, 
           edfType='int')


studyDateTime = data_hrate.index[0]

while(data_hrate.shape[0] % int(data_hr[0][1]) != 0):
    cuoc = data_hrate.shape[0] % int(data_hr[0][1])
    data_hrate.drop(data_hrate.tail(cuoc).index,inplace=True)

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_hrate, 'HR', 'XXXX', sampleRate=int(data_hr[0][1]), 
           units=None, edfType='int')


studyDateTime = data_t.index[0]

while(data_t.shape[0] % int(data_temp[0][1]) != 0):
    cuoc = data_t.shape[0] % int(data_temp[0][1])
    data_t.drop(data_t.tail(cuoc).index,inplace=True)

makeEdf(savePath, 
           savePath.split('/')[-1], studyDateTime, 
           data_t, 'Temperature', 'deg C', sampleRate=int(data_temp[0][1]), 
           units=None, edfType='int')
