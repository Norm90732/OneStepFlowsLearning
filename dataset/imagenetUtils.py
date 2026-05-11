import hydra
from omegaconf import DictConfig
from pathlib import Path 
import os 
import json 
from glob import glob 
import random

def createDirectoryToClassMapping(cfg:DictConfig) -> None:
    trainDir = Path(cfg.dataset.dataDir) / "train"
    
    classDirectories = []
    
    for path in trainDir.iterdir():
        if path.is_dir() == True: 
            classDirectories.append(path)

    labelPath = Path(cfg.dataset.dataDir) / "labelMapping.json"
    if labelPath.exists() == False: 
        labelPath.touch()
    else: 
        raise FileExistsError("The File Already Exists")
    
    numClassDirs = len(classDirectories)
    registryData = {}
    for x in range(0,numClassDirs):
        registryData[f"{str(x)}"] = classDirectories[x].name
    
    with open(Path(cfg.dataset.dataDir) / "labelMapping.json","w") as f: 
        json.dump(registryData,f)
    
    return None  


def getImagePaths(cfg:DictConfig,trainOrval:str):
    if trainOrval not in ("train","val"):
        raise ValueError("use train or val")
    labelMappingPath = Path(cfg.dataset.dataDir) / "labelMapping.json" 
    if labelMappingPath.exists() == False: 
        raise FileNotFoundError(f"make the file at {labelMappingPath} first")
    
    with open(labelMappingPath,"r") as f: 
        mapData = json.load(f)
        #{"0": "n02032355", "1": "n12336092", "2"
    allData= []
    for classIdx, folderName in mapData.items():
        imagePath = Path(cfg.dataset.dataDir) / f"{trainOrval}/{str(folderName)}"
        for img in imagePath.glob("*.JPEG"):
            allData.append({
                "imagepath":str(img),
                "classidx": int(classIdx)
            })
   
    return allData


if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.dataset = OmegaConf.load("configs/dataset/imagenet21k.yaml")
    #createDirectoryToClassMapping(cfg)  #pyrefly:ignore 
    allData = getImagePaths(cfg,"train")
    print(allData[1000000])