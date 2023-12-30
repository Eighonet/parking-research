import os
import datetime

directory = r'annotating/T10LOT/img'
extensions = (['.jpg', '.jpeg', '.png']);

filelist = os.listdir(directory)

newfilesDictionary = {}

count = 0

for file in filelist:
    filename, extension = os.path.splitext(file)
    if ( extension in extensions ):
        create_time = os.path.getmtime( os.path.join(directory, file) )
        format_time = datetime.datetime.fromtimestamp( create_time )
        print(format_time)
        format_time_string = format_time.strftime("%H.%M.%S_%Y-%m-%d")
        newfile = format_time_string + extension;

        if ( newfile in newfilesDictionary.keys() ):
            index = newfilesDictionary[newfile] + 1;
            newfilesDictionary[newfile] = index;
            newfile = format_time_string + '-' + str(index) + extension;
        else:
            newfilesDictionary[newfile] = 0;

        os.rename( os.path.join(directory, file), os.path.join(directory, newfile))
        count = count + 1
        print( file.rjust(35) + '    =>    ' + newfile.ljust(35) )


print( 'All done. ' + str(count) + ' files are renamed. ')