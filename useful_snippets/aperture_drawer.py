def apdrawer_highlight():
    # source: https://stackoverflow.com/questions/51601272/python-matplotlib-heatmap-colorbar-from-transparent
    from matplotlib.colors import LinearSegmentedColormap

    # get colormap
    ncolors = 11
    color_array = plt.get_cmap('tab20b')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='aperture',colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)


def apdrawer(limitc):
    intgrid=limitc*1
    down=[];up=[];left=[];right=[]
    for i, eachline in enumerate(intgrid):
        for j, each in enumerate(eachline):
            if each==1:
                down.append([[j,j+1],[i,i]])
                up.append([[j,j+1],[i+1,i+1]])
                left.append([[j,j],[i,i+1]])
                right.append([[j+1,j+1],[i,i+1]])

    together=[]
    for each in down: together.append(each)
    for each in up: together.append(each)
    for each in left: together.append(each)
    for each in right: together.append(each)

    filtered=[]
    for each in together:
        c=0
        for EACH in together:
            if each==EACH:
                c+=1
        if c==1:
            filtered.append(each)
##########
# Usage: #
##########
plt.figure(figsize=(10,5))
plt.imshow( img, cmap='gray')
plt.imshow( segm.data>0,cmap='aperture',alpha=0.4)
plt.show()

plt.figure(figsize=(10,5))
plt.imshow( img, cmap='gray')

filtered=apdrawer((segm.data>0)*1)
for x in range(len(filtered)):
    plt.plot(np.asarray(filtered[x][0])-0.5,np.asarray(filtered[x][1])-0.5,c='r',linewidth=12)
    plt.plot(np.asarray(filtered[x][0])-0.5,np.asarray(filtered[x][1])-0.5,c='w',linewidth=6)
plt.show()
