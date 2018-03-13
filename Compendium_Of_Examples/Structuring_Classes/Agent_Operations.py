#

from PIL import Image
import numpy as np


from PIL import ImageFilter, ImageMath, ImageChops
import math as math

class ImageOperations:

 def __init__(self, imgs):
     self.imgs = imgs

 def compCandidate(self):
     pass

 def isValid(self):
     pass

 def isRowValid(self):
     pass

 def similarity(self, im1, im2):
     im1 = im1.convert(mode='1')
     im2 = im2.convert(mode='1')
     p_a = list(im1.getdata())
     #print(p_a)
     p_b = list(im2.getdata())
     #print(p_b)
     ab = list(np.absolute(np.array(p_a)) - np.absolute(np.array(p_b)))
     #print(ab)
     #print(sum(ab))
     err = abs(float(sum(ab) / 255)/len(ab))
     return 1 - err

 def getImgVal(self, imgs):
      vals = []
      for i in range(0,len(imgs)):
          img1 = self.getArr(imgs[i])
          vals.append(np.sum(img1)/255)
      return vals


 def getArr(self, img):
     (width, height) = img.size
     o1_arr = np.array(list(img.getdata())).reshape((height, width))
     return o1_arr

 def getEdgeOnlyRow(self, imgs):
      out = []
      for img in imgs:
        out.append(img.filter(ImageFilter.FIND_EDGES))
      return out

 def getEdgeOnlyBlock(self,frames):
     out = []
     for i in range(0,len(frames)):
        out.append(self.getEdgeOnlyRow(frames[i]))
     return out

 def getEdgeOnlyChoices(self, dictOfImgs):
    out = dictOfImgs.copy()
    for key in dictOfImgs:
        out[key]=dictOfImgs[key].filter(ImageFilter.FIND_EDGES)
    return out

 def getDiagonalMatrix(self,imgs, setImgs, modFactor=3):
    out = imgs[:]
    rowSet = self.getImgVal(setImgs)
    row = self.getImgVal(imgs)
    if rowSet == row:
        pass
    else:
        for i in range(0,len(imgs)):
            out[i]=imgs[i-1]
    return out

 def getColImgs(self,objs,colIdx):
    out = []
    for i in range(0, len(objs)):
       try:
           out.append(objs[i][colIdx])
       except:
           break

    return out

 def getOrder(self,nums):
        out = nums[:]
        ordered = sorted(nums)
        for i in range(0,len(nums)):
            for j in range(0,len(ordered)):
                if nums[i]==ordered[j]:
                    out[i]=j
        return out

 def getSegments(self,img, rows, cols):
    (width, height) = img.size
    rowInc = float(height/rows)
    colInc = float(width/cols)
    out = []

    for j in range(0,rows):
        for i in range(0,cols):
            left = int(round(colInc*i))
            right = int(round(colInc*(i+1)))
            top = int(round(rowInc*j))
            bottom = int(round(rowInc*(j+1)))
            temp = img.crop(box=(left,top, right, bottom))
            out.append(temp)
    return out

 def getBlackAndWhiteRow(self,imgs):
     out = []
     for i in range(0,len(imgs)):
         tmp = imgs[i].convert(mode='1')
         out.append(tmp)
     return out

class noOp(ImageOperations):

  #def __init__(self, im1, im2):
  #  super(ImageOperations,self).__init__(im1,im2)
  def is_valid(self, im1, im2, thresh=0.99):
      out = False
      if self.similarity(im1, im2) >= thresh:
          out= True
      return out

  def isValid(self, imgRow, thresh=0.99):
      out = True
      for i in range(0,len(imgRow)-1):
          img1 = imgRow[i]
          img2 = imgRow[i+1]
          if self.similarity(img1,img2) < thresh:
              out = False
              break
      return out

  def getFillFactorRow(self,imgs):
      return imgs

  def compCandidate(self, imgs, choice):
      out = False
      newRow = imgs[:]
      newRow.append(choice)
      if self.isValid(newRow):
          out = True
      return out


class sameSetOp(noOp):
    def getFillFactorRow(self, imgs,setImgs):
        rowValsSet = self.getImgVal(setImgs)
        rowSet = sorted(rowValsSet)
        rowVals = self.getImgVal(imgs)
        row1 = sorted(rowVals)
        return [rowSet,row1]

    def isValid(self, rows, thresh=0.99):
        out = True
        for i in range(0, len(rows[0])):
            val = 1-float(abs(rows[0][i]-rows[1][i]))/min(rows[0][i],rows[1][i])

            if val<=thresh:
                out = False
                break
        return out

    def compCandidate(self, imgs, choice, setImgs):
          out = False
          newRow = imgs[:]
          newRow.append(choice)
          rows = self.getFillFactorRow(newRow,setImgs)
          if self.isValid(rows):
              out = True
          return out


class moveOp(noOp):

  #def __init__(self, im1, im2):
  #  super(ImageOperations,self).__init__(im1,im2)

  def getFillFactor(self,im1):
      #arr1 = self.getArr(im1)
      #out =  np.sum(arr1)/255
      width,height = im1.size
      area = width * height
      arr1 = self.getArr(im1)
      out =  float(np.sum(arr1)/255)/area

      return out

  def getFillFactorRow(self, imgs):
    diffs = []
    for i in range(0,len(imgs)):
      img1 = imgs[i]
      diffs.append(self.getFillFactor(img1))
    return diffs

  def getFillFactorBlock(self,frames):
    out = []
    for i in range(0,len(frames)):
        temp = self.getFillFactorRow(frames[i])
        out.append(temp)
    return out

  def isSegmentSame(self, factorRow, segmentInd, frameInd, thresh=0.015):
      out = False
      print(abs(factorRow[segmentInd[0]][frameInd[0]] - factorRow[segmentInd[1]][frameInd[1]]))
      if abs(factorRow[segmentInd[0]][frameInd[0]] - factorRow[segmentInd[1]][frameInd[1]]) <= thresh:
          out = True
      return out

  def isSegmentSameBlock(self,frames, segmentInd, frameInd, thresh=6):
      out = True
      for i in range(0,len(frames)):
        if self.isSegmentSame(frames[i],segmentInd,frameInd,thresh):
          out = False
          break
      return out

class fillOp(ImageOperations):

      def getDiffArray(self, im1, im2):
          (width, height) = im1.size
          p_a = self.getArr(im1)
          p_b = self.getArr(im2)
          ab = np.resize((p_a-p_b),[width, height])
          return ab

      def getDiffHistogram(self, im1, im2):
          diff = ImageChops.difference(im1,im2).getdata()
          out =  np.sum(diff)/255
          return out

      def isValid(self, factorRow,thresh=.01):
          out = True
          for i in range(0,len(factorRow)-1):
            if float((abs(factorRow[i+1]-factorRow[i]))/float(max(abs(factorRow[i]),abs(factorRow[i+1])))) >= thresh:
                out = False
                break
          print(out)
          return out

      def getFillFactorRowHist(self,imgs):
          out = []
          for i in range(0, len(imgs)-1):
              out.append(self.getDiffHistogram(imgs[i],imgs[i+1]))
          return out

      def getFillFactor(self, im1, im2):
          width, height = im1.size
          div = width * height
          size1 = float(np.sum(abs(self.getArr(im1)))/255)
          size2 = float(np.sum(abs(self.getArr(im2)))/255)
          fillFactor = size2/size1
          #print(fillFactor)
          return fillFactor

      def getFillFactorRow(self, imgs, thresh=100):
          factors = []
          for i in range(0,len(imgs)-1):
              img1 = imgs[i]
              img2 = imgs[i+1]
              factors.append(self.getFillFactor(img1, img2))
          return factors

      def compCandidate(self, imgs, choice, thresh=100):
          out = False
          row = imgs[:]
          row.append(choice)

          choiceFactor = self.getFillFactorRow(row)
          print("choice factor")
          print(choiceFactor)
          if self.isValid(choiceFactor) == True:
              out = True
          return out


class transformOp(fillOp):
      def getFillFactor(self, im1, im2):
          p_a = self.getArr(im1)
          p_b = self.getArr(im2)
          pixelDiff = (np.sum(p_b)-np.sum(p_a))/255

          return pixelDiff

      def getFillFactorRow(self, imgs):
          diffs = []
          for i in range(0,len(imgs)-1):
              img1 = imgs[i]
              img2 = imgs[i+1]
              diffs.append(self.getFillFactor(img1, img2))
          print('diffs')
          print(diffs)
          return diffs

      def isValid(self, factorRow,thresh=10):
          out = True
          for i in range(0,len(factorRow)-1):
            if abs(factorRow[i+1]-factorRow[i]) >= thresh:
                out = False
                break
          print('isValid?')
          print(out)
          return out


      def isOneDirection(self,factorRow):
        out = False
        row = np.array(factorRow)
        if (row>0).all(axis=0) or (row<0).all(axis=0):
            out = True
        return out

      def compCandidate(self, imgs, choice):
          out = False
          row = imgs[:]
          row.append(choice)
          #row = self.getEdgeOnlyRow(row)
          '''
          for i in range(0, len(row)):
              row[i].save(str(i),'JPEG')
          '''

          pixelFactor = self.getFillFactorRow(row)
          print("pixel factor")
          print(pixelFactor)
          if self.isValid(pixelFactor) == True:
              out = True
          return out

class transformOpBySet(transformOp):
      def getFillFactorRow(self, imgs, setImgs):
          diffs = transformOp.getFillFactorRow(self,imgs)
          diffsSet = transformOp.getFillFactorRow(self,setImgs)
          diffSq = np.array(diffs)-np.array(diffsSet)
          diffSq = diffSq.tolist()
          print('diffSq')
          print(diffSq)
          return diffSq

      def isValid(self, diffsRow,thresh=10):
          out = True
          for i in range(0,len(diffsRow)):
            print('difference')
            print(abs(diffsRow[i]))
            if abs(diffsRow[i]) > thresh:
                out = False
                break
          print('isValid?')
          print(out)
          return out

      def compCandidate(self, imgs, choice,setImgs):
          out = False
          row = imgs[:]
          row.append(choice)
          #row = self.getEdgeOnlyRow(row)
          pixelFactor = self.getFillFactorRow(row, setImgs)
          print("pixel factor")
          print(pixelFactor)
          if self.isValid(pixelFactor) == True:
              out = True
          return out

class transformOpBySetConstantDiff(transformOpBySet):
      def getFillFactorRow(self, imgs, setImgs):
          rowValsSet = self.getImgVal(setImgs)
          rowSet = sorted(rowValsSet)
          rowVals = self.getImgVal(imgs)
          row1 = sorted(rowVals)
          diffs = np.array(rowSet)-np.array(row1)
          diffSq = diffs.tolist()
          print('row of differences')
          print(diffSq)
          return diffSq

      def isValid(self, diffsRow,thresh=20):
          out = True
          diffsRowNP = np.array(diffsRow)
          if (diffsRowNP==0).all(axis=0):
              pass
          else:
              factor = diffsRow[0]
              for i in range(0,len(diffsRow)):
                print('diff!!!!')
                print(abs(diffsRow[i]-factor))
                if abs(diffsRow[i]-factor) > thresh:
                    out = False
                    break
          print('isValid?')
          print(out)
          return out

class transformOpBySetDiag(transformOpBySetConstantDiff):

    def subOrdered(self,refImgs):
        refVal = self.getImgVal(refImgs)
        order = self.getOrder(refVal)
        return order

    def checkSameOrder(self,order1,refOrder):
        out =False
        for i in range(0,len(order1)):
            temp = order1[:]
            for a in range(0,len(order1)):
                temp[a] = order1[a-i]
                #temp[a] = order1[divmod(a+i,len(order1))[1]]
            if temp == refOrder:
                out = True
                break
        return out



    def getFillFactorRow(self, imgs, refImgs):
      imgOrder = self.subOrdered(imgs)
      refOrder = self.subOrdered(refImgs)
      return [imgOrder,refOrder]

    def isValid(self, listOfOrders, thresh = 20):
        orderCheck = self.checkSameOrder(listOfOrders[0],listOfOrders[1])
        print('isTrue?')
        print(orderCheck)
        return orderCheck

    def compCandidate(self, imgs, choice,refImgs):
          out = False
          row = imgs[:]
          row.append(choice)
          order = self.getFillFactorRow(row, refImgs)
          print("choice order")
          print(order)
          if self.isValid(order) == True:
              out = True
          return out
class divideImage(ImageOperations):

    def getSegments(self,img, rows, cols):
        (width, height) = img.size
        rowInc = float(height/rows)
        colInc = float(width/cols)
        out = []

        for j in range(0,rows):
            for i in range(0,cols):
                left = int(round(colInc*i))
                right = int(round(colInc*(i+1)))
                top = int(round(rowInc*j))
                bottom = int(round(rowInc*(j+1)))
                temp = img.crop(box=(left,top, right, bottom))
                out.append(temp)
        return out

    def groupSegments(self, imgs, rows, cols):
        storage = []
        for i in range(0,len(imgs)):
            temp = self.getSegments(imgs[i],rows, cols)
            storage.append(temp)
        out = []
        for i in range(0,(rows*cols)):
            temp = []
            for j in range(0,len(imgs)):
                temp.append(storage[j][i])
            out.append(temp)
        return out

    def isRowValid(self, opObj, objs, thresh):
       flag = True
       for i in range(0,len(objs)):
           print("baselines")
           #objs[i]=transObj.getEdgeOnlyRow(objs[i])
           temp  = opObj.getFillFactorRow(objs[i])
           print(temp)
           #flag = opObj.isValid(temp,thresh)
       return flag

    def compCandidate(self, imgs, choice,refImgs):
          out = False
          row = imgs[:]
          row.append(choice)
          order = self.getFillFactorRow(row, refImgs)
          print("choice order")
          print(order)
          if self.isValid(order) == True:
              out = True
          return out

class blendImgOp(transformOpBySetDiag):

    def blendImgbyBlack(self,im1,im2):
        im = im1.copy()
        im1 = im1.convert(mode='1')
        im2 = im2.convert(mode='1')
        (width,height) = im.size
        for w in range(0,width):
            for h in range(0, height):
                x = im1.getpixel((w,h))
                y = im2.getpixel((w,h))
                combPix = x + y
                if combPix==255:
                    combPix =  255
                elif combPix==510:
                    combPix = 255
                im.putpixel((w,h),combPix)
        return im

    def blendImgbyWhite(self,im1,im2):
        im = im1.copy()
        im1 = im1.convert(mode='1')
        im2 = im2.convert(mode='1')
        (width,height) = im.size
        for w in range(0,width):
            for h in range(0, height):
                x = im1.getpixel((w,h))
                y = im2.getpixel((w,h))
                combPix = x + y
                if combPix==510:
                    combPix = 255
                im.putpixel((w,h),combPix)
        return im

    def blendImgbyNumBlack(self,im1, im2):
        im1 = im1.convert(mode='1')
        im2 = im2.convert(mode='1')
        blackArea = self.getImgVal([im2])
        (weight, height)=im2.size
        area = weight * height
        blackArea = area - blackArea[0]
        orig = self.getImgVal([im1])
        blendArea = orig[0]+blackArea
        return blendArea

    def numImageRow(self,imgs):
        out = []
        newImgs = imgs[:]
        for i in range(0,len(newImgs)):
            newImgs[i]= newImgs[i].convert(mode='1')
            (width,height)=newImgs[i].size
            area = width*height
            val = self.getImgVal([newImgs[i]])[0]
            out.append(area - val)
        return out

    def isTopHeavy(self,img):
        segments = self.getSegments(img,2,1)
        (width, height)=img.size
        area = width * height
        valTop = self.getImgVal([segments[0]])[0]
        blackAreaTop = area - valTop
        valBottom = self.getImgVal([segments[1]])[0]
        blackAreaBottom = area - valBottom
        if blackAreaTop > blackAreaBottom:
            out = True
        else:
            out = False
        return out


    def blendImgbyOverlapBlack(self,im1,im2):
        im = im1.copy()
        im1 = im1.convert(mode='1')
        im2 = im2.convert(mode='1')
        (width,height) = im.size
        for w in range(0,width):
            for h in range(0, height):
                x = im1.getpixel((w,h))
                y = im2.getpixel((w,h))
                combPix = x + y
                if combPix==255:
                    combPix = 0
                elif combPix == 510:
                    combPix = 255
                elif combPix == 0:
                    combPix = 255
                im.putpixel((w,h),combPix)
        return im


    def blendImgbyBlack2(self,im1,im2):
        im = im1.copy()
        im1 = im1.convert(mode='1')
        im2 = im2.convert(mode='1')
        (width,height) = im.size
        for w in range(0,width):
            for h in range(0, height):
                x = im1.getpixel((w,h))
                y = im2.getpixel((w,h))
                combPix = x + y
                if combPix==255:
                    combPix =  0
                elif combPix==510:
                    combPix = 255
                im.putpixel((w,h),combPix)
        return im

    def getFirstLastImg(self,row):
        return [row[0],row[2]]

    def getMiddleLastImg(self,row):
        return [row[1],row[2]]

    def getFillFactorRow(self, imgs, blendFcn):
        #Assumes 3 images are being passed
        blend = blendFcn(imgs[0],imgs[1])
        out = [blend,imgs[2]]
        return out

    def sortImgsByBlack(self,imgs):
        rowVal = self.getImgVal(imgs)
        rowDict = dict(zip(rowVal,imgs))
        print('rowdict')
        print(rowDict)
        rowKeySorted = sorted(rowVal,reverse=True)
        print('rowKeySorted')
        print(rowKeySorted)
        out = []
        for i in rowKeySorted:
            out.append(rowDict[i])

        return out

    def getFillFactorRowSort(self, imgs, blendFcn):
        #Assumes 3 images are being passed
        rowVal = self.getImgVal(imgs)
        rowDict = dict(zip(rowVal,imgs))
        print('rowdict')
        print(rowDict)
        rowKeySorted = sorted(rowVal,reverse=True)
        print('rowKeySorted')
        print(rowKeySorted)
        blend = blendFcn(rowDict[rowKeySorted[0]],rowDict[rowKeySorted[1]])
        blend.save('L:\OMSCS\KBAI\Project-Code-Python\Problems\sample.jpeg','JPEG')
        out = [blend,rowDict[rowKeySorted[len(rowKeySorted)-1]]]
        return out


    def getFillFactorRowMiddleLast(self, imgs, blendFcn):
        #Assumes 3 images are being passed
        blend = blendFcn(imgs[0],imgs[2])
        out = [blend,imgs[1]]
        return out

    def getFillFactorRowOnTopHalf(self, imgs, blendFcn):
        #Assumes 3 images are being passed

        topHalves = []
        for i in range(0,len(imgs)):
            segments = self.getSegments(imgs[i],2,1)[1]
            topHalves.append(segments)

        return self.getFillFactorRow(topHalves,blendFcn)

    def getFillFactorRowForSeparateHalves(self, imgs, blendFcn):
        #Assumes 3 images are being passed

        topHalves = []
        for i in range(0,len(imgs)):
            segments = self.getSegments(imgs[i],2,1)[0]
            topHalves.append(segments)

        bottomHalves = []
        for i in range(0,len(imgs)):
            segments = self.getSegments(imgs[i],2,1)[1]
            bottomHalves.append(segments)

        topRow =  self.getFirstLastImg(topHalves)
        bottomRow = self.getMiddleLastImg(bottomHalves)

        return [topRow,bottomRow]

    def getFillFactorQE12(self, imgs, blendFcn='filler'):
        #Assumes 3 images are being passed
        imgVals = self.numImageRow(imgs)
        print('areas')
        print(imgVals)
        errorPct = float(abs(imgVals[0]-imgVals[1]-imgVals[2]))/imgVals[0]

        img0TopHeavy = self.isTopHeavy(imgs[0])
        img2TopHeavy = self.isTopHeavy(imgs[2])


        return [errorPct, img0TopHeavy==img2TopHeavy]

    def isValid(self,twoImgs,threshold=0.99):
        out = False
        print('simiarlity')
        print(self.similarity(twoImgs[0],twoImgs[1]))
        if self.similarity(twoImgs[0],twoImgs[1]) >=threshold:
            out = True
        return out

    def isValid_imgSimilarity(self,twoImgs,threshold=0.99):
        out = False
        print('simiarlity')
        print(self.imgSimilarity(twoImgs[0],twoImgs[1]))
        if self.imgSimilarity(twoImgs[0],twoImgs[1]) >=threshold:
            out = True
        return out

    def isValid_TwoRows_Comp_FirstLast(self,twoRows,threshold=0.99):
        out = True
        for row in twoRows:
            print('similarity')
            print(self.similarity(row[0],row[1]))
            if self.similarity(row[0],row[1]) < threshold:
                out = False
                break
        return out

    def isValid_byNumBlack(self,comps,threshold=0.01):
        out = False
        realArea = self.getImgVal([comps[1]])[0]
        print('real area')
        print(realArea)
        print('blended area')
        print(comps[0])
        if float(abs(comps[0]-realArea))/realArea <=threshold:
            out = True
        return out

    def isValidQE12(self,input,threshold=0.01):
        out = False
        if input[0] <= threshold:
            if input[1] == True:
                out = True
        return out

    def imgSimilarity(self,im1,im2):
        (width, height) = im1.size
        countImg = im1.copy()
        img1 = im1.convert(mode='1')
        img2 = im2.convert(mode='1')
        for i in range(0,width):
            for j in range(0,height):
                p1 = img1.getpixel((i,j))
                p2 = img2.getpixel((i,j))
                if p1==p2:
                    countImg.putpixel((i,j),255)
                else:
                    countImg.putpixel((i,j),0)
        similarity = self.getImgVal([countImg])
        similarity = float(similarity[0])/(width*height)

        return similarity

    def compCandidate(self, imgs, choice, thresh,fillRowFcn,blendFcn,validFcn):
          out = False
          row = imgs[:]
          row.append(choice)
          order = fillRowFcn(row,blendFcn)
          print("blended image")
          print(order)
          if validFcn(order,thresh) == True:
              out = True
              print('this one is true')
          return out

class answerOp(transformOp):

    def getAnswerRows(self, ans, given, choices):
        out = []
        for a in ans:
            temp = given[:]
            temp.append(choices[a])
            out.append(temp)
        return out

    def elimByPixels(self,answerDict):
        out = answerDict.copy()
        for a in answerDict:
            factorRow = self.getFillFactorRow(out[a])
            print(factorRow)
            factorRow = np.array(factorRow)

            if (factorRow>0).all(axis=0) or (factorRow<=0).all(axis=0):
                pass
            else:
                del out[a]

        return out


    def elimBySizeOrder(self,answerDict,refImgs):
        out = answerDict.copy()
        refVal = self.getImgVal(refImgs)
        refOrder = self.getOrder(refVal)
        print('ref order')
        print(refOrder)
        for a in answerDict:
            aVal = self.getImgVal(answerDict[a])
            rowOrder = self.getOrder(aVal)
            print('rowOrder')
            print(a)
            print(rowOrder)

            if rowOrder!=refOrder:
                del out[a]
        print('out')
        print(out)
        return out

    def elimByNoDuplicates(self,answerDict,refImgs,thresh=0.0005):
        out = answerDict.copy()
        refVal = self.getImgVal(refImgs)
        refOrdered = sorted(refVal)
        refDupl = True
        for i in range(0,len(refOrdered)-1):
            if float(abs(refOrdered[i]-refOrdered[i+1]))/refOrdered[i] <= thresh:
                refDupl = False
                break

        print('ref order')
        print(refOrdered)
        print('gooon?')
        print(refDupl)
        if refDupl:
            for a in answerDict:
                aVal = self.getImgVal(answerDict[a])
                rowOrdered = sorted(aVal)
                print('rowOrder')
                print(a)
                print(rowOrdered)
                for i in range(0,len(rowOrdered)-1):
                    print('ratio')
                    print(float(abs(rowOrdered[i]-rowOrdered[i+1]))/rowOrdered[i])
                    if float(abs(rowOrdered[i]-rowOrdered[i+1]))/rowOrdered[i] <= thresh:
                        del out[a]
                        break


        return out

    def elimByImgByNumBlack(self,answerDict,colImgs,refImgs,thresh, blendFcn, validFcn, fillRowFcn):
        out = answerDict.copy()
        refRow = fillRowFcn(refImgs,blendFcn)
        goOn = validFcn(refRow,thresh)
        print('goOn')
        print(goOn)
        if goOn:
            for a in answerDict:
                newChoice = colImgs[:]
                newChoice.append(answerDict[a][2])
                row = fillRowFcn(newChoice,blendFcn)
                print('row___True??')
                print(validFcn(row,thresh))
                if validFcn(row,thresh) == False:
                    del out[a]
        return out

    def elimBySimilarity(self,answerDict,thresh=0.05):
        out = answerDict.copy()
        for a in answerDict:
            factorRow = self.getFillFactorRow(out[a])
            print(factorRow)
            factorRow = np.array(factorRow)
            print('factor')
            print(float(abs(factorRow[len(factorRow)-1] - factorRow[len(factorRow)-2]))/abs(factorRow[len(factorRow)-1]))

            if float(abs(factorRow[len(factorRow)-1] - factorRow[len(factorRow)-2]))/max(abs(factorRow[len(factorRow)-1]),abs(factorRow[len(factorRow)-2])) <= thresh :
                pass
            else:
                del out[a]

        return out

    def elimByImgSimilarityTopHalf(self,answerDict,thresh=.90):
        out = answerDict.copy()
        blendObj = blendImgOp('filler')
        for a in answerDict:
            seg1 = self.getSegments(answerDict[a][0],2,1)[0]
            seg2 = self.getSegments(answerDict[a][2],2,1)[0]

            sim = blendObj.imgSimilarity(seg1,seg2)
            print('sim')
            print(sim)
            if sim <= thresh:
                del out[a]
        print('out')
        print(out)
        return out



    def elimByFactor(self,answerDict,factor,thresh=0.02):
        out = answerDict.copy()
        print('checkFactor')
        print(factor)
        for a in answerDict:
            print('checking number')
            print(a)
            factorRow = self.getImgVal(out[a])
            print(factorRow)
            factorRow = np.array(factorRow)
            print('factor')
            myFactor = abs(float(factorRow[len(factorRow)-1])/factorRow[len(factorRow)-2])
            print(myFactor)
            if float(abs(myFactor - factor))/factor <= thresh :
                pass
            else:
                del out[a]

        return out

    def verticalReflection(self, img):
        img_dest = img.copy()
        (width,height)=img.size
        for x in range(0,width):
            for y in range(0,height):
                p = img.getpixel((x,y))
                img_dest.putpixel(((width/2-1+x)%width,y),p)
        img_dest.save("mirroor",'JPEG')
        return img_dest

    def imgCompare(self, img1,img2):
        img_dest = img1.copy()
        (width,height)=img1.size
        count  = 0
        for x in range(0,width):
            for y in range(0,height):
                p1 = img1.getpixel((x,y))
                p2 = img2.getpixel((x,y))
                if p1==p2:
                    count = count+1
        out = float(count) / (width*height)
        return out

    def elimByVerticalReflection(self,answerDict, compIdx, thresh=0.9):
        out = answerDict.copy()
        for a in answerDict:
            compImg = answerDict[a][compIdx]
            newImg = self.verticalReflection(answerDict[a][len(answerDict[a])-1])
            ratio = self.imgCompare(compImg,newImg)
            if ratio < thresh :
                del out[a]
        return out
    def countPixelsFirstColumn(self,img):
        (width,height)=img.size
        count  = 0
        goOn = True
        for x in range(0,width):
            if goOn == False:
                break
            for y in range(0,height):
                state = img.getpixel((x,y))
                if state < 255:
                    for y1 in range(0,height):
                        if img.getpixel((x,y1)) < 255:
                            count = count + 1
                    goOn= False
                    break
        return count

    def elimByFirstColumn(self,answerDict, compIdx, thresh=3):
        out = answerDict.copy()
        for a in answerDict:
            col1 = self.countPixelsFirstColumn(answerDict[a][compIdx])
            col2 = self.countPixelsFirstColumn(answerDict[a][len(answerDict[a])-1])
            diff = abs(col1 - col2)
            if diff > thresh:
                del out[a]
        return out

    def isValid(self, answers, thresh=1000):
        out =  False
        if len(answers) == 1:
            out = True
        return out

class rotateOp(ImageOperations):

    def rotateImg(self, img, deg):
        newImg= img.rotate(deg)
        return newImg

    def rotateRow(self, imgs, deg):
        out = []
        for i in range(0,len(imgs)):
            out.append(self.rotateImg(imgs[i],deg))

        return out



class framesControl():

    def getProbRelation(self,objs,opOP,choices,opFcn,opValid,thresh,*obArg):
        flag = True
        for i in range(0,len(objs)-1):
            temp  = opFcn(objs[i],*obArg)
            flag = opValid(temp,thresh)
            if flag ==False:
                break
        return flag

    def testChoices(self,objs,choices,compFcn,**choiceArgs):
        answers = {}
        for choice in choices:
              print(choice)
              if compFcn(objs[len(objs)-1], choices[choice],**choiceArgs) == True:
                  ans = int(choice)
                  candidate = objs[len(objs)-1][:]
                  candidate.append(choices[choice])
                  answers[ans]=candidate
        return answers

    def elimByFcn(self,answerDict, fcn,**args):
        return fcn(answerDict,**args)

