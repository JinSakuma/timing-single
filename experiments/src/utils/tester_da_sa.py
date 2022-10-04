import torch
import numpy as np
import os
from tqdm import tqdm
import wandb


    
def val(model, data_loader, deivce):
    model.eval()
    
    total = {"asr_cer": 0, "loss":0,
             "dialog_p": 0, "dialog_r": 0, "dialog_f1": 0,
             "dialog_correct": 0, "dialog_total": 0,
             "system_p": 0, "system_r": 0, "system_f1": 0,
             "system_correct": 0, "system_total": 0,
            }
    ccounter = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = model(batch, "val")
            
            if outputs["val_loss"] != 0:
                total["loss"] += outputs["val_loss"].detach().cpu().numpy()
            else:
                total["loss"] += 0
            
            total["dialog_p"] += outputs["val_dialog_acts_precision"]
            total["dialog_r"] += outputs["val_dialog_acts_recall"]
            total["dialog_f1"] += outputs["val_dialog_acts_f1"]
            total["dialog_correct"] += outputs["val_dialog_acts_acc"]
            total["dialog_total"] += outputs["val_num_dialog_acts_total"]
            total["system_p"] += outputs["val_system_acts_precision"]
            total["system_r"] += outputs["val_system_acts_recall"]
            total["system_f1"] += outputs["val_system_acts_f1"]
            total["system_correct"] += outputs["val_system_acts_acc"]
            total["system_total"] += outputs["val_num_system_acts_total"]
            total["asr_cer"] += outputs["val_asr_cer"]
            ccounter += 1
            
        dialog_p = float(total["dialog_p"]) / ccounter
        dialog_r = float(total["dialog_r"]) / ccounter
       	#dialog_f1 = float(total["dialog_f1"]) / ccounter
        dialog_f1 = 2 * dialog_p * dialog_r / float(dialog_p + dialog_r) if (dialog_p + dialog_r) != 0 else 0
        dialog_acc = float(total["dialog_correct"]) / total["dialog_total"] if total["dialog_total"]>0 else 0
        system_p = float(total["system_p"]) / ccounter
        system_r = float(total["system_r"]) / ccounter
       	#system_f1 = float(total["system_f1"]) / ccounter
        system_f1 = 2 * system_p * system_r / float(system_p + system_r) if (system_p + system_r) != 0 else 0
        system_acc = float(total["system_correct"]) / total["system_total"] if total["system_total"]>0 else 0
       	cer = float(total["asr_cer"]) / ccounter 
        loss = float(total["loss"]) / ccounter

    return loss, cer, (dialog_p, dialog_r, dialog_f1, dialog_acc), (system_p, system_r, system_f1, system_acc)

def tester(model, loader_dict, optimizer, device, outdir, is_use_wandb):
	
	val_loss, val_cer, val_dialog, val_system = val(model, loader_dict["val"], device)
	val_dialog_p, val_dialog_r, val_dialog_f1, val_dialog_acc = val_dialog
	val_system_p, val_system_r, val_system_f1, val_system_acc = val_system

	print("Val loss: {}".format(val_loss))
	print("Val WER: {}".format(val_cer))
	print("Val DA Precision: {}".format(val_dialog_p))
	print("Val DA Recall: {}".format(val_dialog_r))
	print("Val DA F1: {}".format(val_dialog_f1))
	print("Val DA Acc: {}".format(val_dialog_acc))    
	print("Val SA Precision: {}".format(val_system_p))
	print("Val SA Recall: {}".format(val_system_r))
	print("Val SA F1: {}".format(val_system_f1))
	print("Val SA Acc: {}".format(val_system_acc))        
	
	test_loss, test_cer, test_dialog, test_system = val(model, loader_dict["test"], device)
	test_dialog_p, test_dialog_r, test_dialog_f1, test_dialog_acc = test_dialog
	test_system_p, test_system_r, test_system_f1, test_system_acc = test_system

	print("Test loss: {}".format(test_loss))
	print("Test WER: {}".format(test_cer))
	print("Tset DA Precision: {}".format(test_dialog_p))
	print("Test DA Recall: {}".format(test_dialog_r))
	print("Test DA F1: {}".format(test_dialog_f1))
	print("Test DA Acc: {}".format(test_dialog_acc))  
	print("Test SA Precision: {}".format(test_system_p))
	print("Test SA Recall: {}".format(test_system_r))
	print("Test SA F1: {}".format(test_system_f1))
	print("Test SA Acc: {}".format(test_system_acc))    
