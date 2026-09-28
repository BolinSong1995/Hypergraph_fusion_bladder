from __future__ import annotations

import numpy as np
import torch

from .utils import cox_ph_loss, harrell_c_index


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    risks, times, events, patient_ids = [], [], [], []

    for sample in loader:
        ct = sample["ct"].to(device)
        wsi = sample["wsi"].to(device)

        risk = model(ct, wsi)

        risks.append(float(risk.detach().cpu()))
        times.append(float(sample["time"]))
        events.append(int(sample["event"]))
        patient_ids.append(sample["patient_id"])

    cindex = harrell_c_index(risks, times, events)
    return cindex, risks, times, events, patient_ids


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    cox_batch_size=16,
    require_event_in_batch=True,
    grad_clip_norm=5.0,
):
    model.train()

    epoch_losses = []
    train_risks, train_times, train_events = [], [], []

    acc_risk, acc_time, acc_event = [], [], []

    for batch_idx, sample in enumerate(loader):
        ct = sample["ct"].to(device)
        wsi = sample["wsi"].to(device)
        time = sample["time"].to(device).view(1)
        event = sample["event"].to(device).view(1)

        risk = model(ct, wsi).view(1)

        train_risks.append(float(risk.detach().cpu()))
        train_times.append(float(time.detach().cpu()))
        train_events.append(int(event.detach().cpu()))

        acc_risk.append(risk)
        acc_time.append(time)
        acc_event.append(event)

        enough_patients = len(acc_risk) >= cox_batch_size
        has_event = (
            torch.cat(acc_event).sum().item() > 0
            if acc_event
            else False
        )
        last_batch = batch_idx == len(loader) - 1

        should_update = (
            enough_patients
            and (not require_event_in_batch or has_event)
        )

        if last_batch and len(acc_risk) > 0:
            should_update = (
                not require_event_in_batch or has_event
            )

        if should_update:
            risks = torch.cat(acc_risk, dim=0)
            times = torch.cat(acc_time, dim=0)
            events = torch.cat(acc_event, dim=0)

            loss = cox_ph_loss(risks, times, events)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            )
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

            acc_risk, acc_time, acc_event = [], [], []

    train_cindex = harrell_c_index(
        train_risks, train_times, train_events
    )
    mean_loss = (
        float(np.mean(epoch_losses)) if epoch_losses else np.nan
    )

    return mean_loss, train_cindex
