from __future__ import division
from pox.core import core  
import pox.openflow.libopenflow_01 as of  
from pox.lib.revent import *  
from pox.lib.recoco import Timer  
from collections import defaultdict  
from pox.openflow.discovery import Discovery  
from pox.lib.util import dpid_to_str  
import time

def mean(numbers):
	return sum(numbers) / len(numbers)

def sqrt(number):
	return number ** (1/2)

def std(numbers):
	avg = mean(numbers)
	accumulator = 0
	for x in numbers:
		accumulator += (x - avg) ** 2
	return sqrt(accumulator/(len(numbers)))


class Flow:
	def __init__(self, src, dst, tp_src, tp_dst):
		self.src = src
		self.dst = dst
		self.tp_src = tp_src
		self.tp_dst = tp_dst
		self.durations = []
		self.bytes = []
		self.packets = []
	
	def add(self, duration, bytes, packets):
		self.durations.append(duration)
		self.bytes.append(bytes)
		self.packets.append(packets)

	def printValues(self):
		print("SRC: {}:{}".format(self.src, self.tp_src))
		print("DST: {}:{}".format(self.dst, self.tp_dst))
		for i in range(len(self.bytes)) :
			print("Iteration: {}".format(i))
			print("Bytes: {}".format(self.bytes[i]))
			print("Packets: {}".format(self.packets[i]))

	def __eq__(self,other):
		return self.src == other.src and self.dst == other.dst and self.tp_src == other.tp_src and self.tp_dst == other.tp_dst

class FullFlow:
	def __init__(self, forward):
		self.forward = forward

	def addBackward(self, backward):
		self.backward = backward
	
	def hasBothFlows(self):
		return hasattr(self, 'forward') and hasattr(self, 'backward')

	def isFlow(self):
		return self.hasBothFlows() and len(self.forward.durations) > 1 and len(self.backward.durations) > 1

	def printValues(self):
		print("-------------------------------------")
		print("{}:{} <-> {}:{}".format(self.forward.src, self.forward.tp_src, self.forward.dst, self.forward.tp_dst))
		print("Readings -> {}".format(len(self.forward.bytes)))
		readingsBackwards = 0
		if hasattr(self, 'backward'):
			readingsBackwards = len(self.backward.bytes)
		print("Readings <- {}".format(readingsBackwards))
		print("Last Duration -> {}".format(self.forward.durations[len(self.forward.durations)-1]))
		if hasattr(self, 'backward'):
			print("Last Duration <- {}".format(self.backward.durations[len(self.backward.durations)-1]))
		forwardBytes = ''
		forwardPackets = ''
		forwardDurations = ''
		backwardBytes = ''
		backwardPackets = ''
		backwardDurations = ''
		for b in self.forward.bytes:
			forwardBytes += str(b) + ' -'
		for p in self.forward.packets:
			forwardPackets += str(p) + ' -'
		for d in self.forward.durations:
			forwardDurations += str(d) + ' -'
		if hasattr(self, 'backward'):
			for b in self.backward.bytes:
				backwardBytes += str(b) + ' -'
			for p in self.backward.packets:
				backwardPackets += str(p) + ' -'
			for d in self.backward.durations:
				backwardDurations += str(d) + ' -'
		print("Bytes ->: {}".format(forwardBytes))
		print("Packets ->: {}".format(forwardPackets))
		print("Duration ->: {}".format(forwardDurations))
		print("Bytes <-: {}".format(backwardBytes))
		print("Packets <-: {}".format(backwardPackets))
		print("Duration <-: {}".format(backwardDurations))
	
	def printStats(self):
		print("-------------------------------------")
		# Flow Duration
		forwardDuration = self.forward.durations[len(self.forward.durations)-1]
		backwardDuration = self.backward.durations[len(self.backward.durations)-1]
		fullDuration = forwardDuration if forwardDuration > backwardDuration else backwardDuration
		# Flow bytes and packets per second
		forwardBytes = self.forward.bytes[len(self.forward.bytes)-1]
		backwardBytes = self.backward.bytes[len(self.backward.bytes)-1]
		forwardPackets = self.forward.packets[len(self.forward.packets)-1]
		backwardPackets = self.backward.packets[len(self.backward.packets)-1]
		fb_psec = (forwardBytes + backwardBytes) / fullDuration
		fp_psec = (forwardPackets + backwardPackets) / fullDuration
		# Forward Inter Arrival Time
		fiat = []
		forwardIdleWindow = 0
		forwardActiveWindow = 0
		forwardIdle = []
		forwardActive = []
		for i in range(len(self.forward.durations)):
			if i > 0:
				duration = self.forward.durations[i] - self.forward.durations[i-1]
				packets = self.forward.packets[i] - self.forward.packets[i-1]
			else:
				packets = self.forward.packets[i]
				duration = self.forward.durations[i]
			if packets > 0:
				# Active on this reading
				currentFiat = duration/packets
				fiat.append(currentFiat)
				forwardActiveWindow += duration
				if forwardIdleWindow > 0:
					# Transition from idle to active
					forwardIdle.append(forwardIdleWindow)
					forwardIdleWindow = 0
			else:
				# Idle on this reading
				forwardIdleWindow += duration
				if forwardActiveWindow > 0:
					# Transition from active to idle
					forwardActive.append(forwardActiveWindow)
					forwardActiveWindow = 0
		# Finish Window if any is open
		if forwardActiveWindow > 0:
			forwardActive.append(forwardActiveWindow)
		if forwardIdleWindow > 0:
			forwardIdle.append(forwardIdleWindow)
		# Add value if no window came to be
		if len(forwardIdle) == 0:
			forwardIdle.append(0)
		if len(forwardActive) == 0:
			forwardActive.append(0)
		# Backward Inter Arrival Time
		biat = []
		backwardIdleWindow = 0
		backwardActiveWindow = 0
		backwardIdle = []
		backwardActive = []
		for i in range(len(self.backward.durations)):
			duration = 0
			packets = 0
			if i > 0:
				duration = self.backward.durations[i] - self.backward.durations[i-1]
				packets = self.backward.packets[i] - self.backward.packets[i-1]
			else:
				packets = self.backward.packets[i]
				duration = self.backward.durations[i]
			if packets > 0:
				# Active on this reading
				currentBiat = duration/packets
				biat.append(currentBiat)
				backwardActiveWindow += duration
				if backwardIdleWindow > 0:
					# Transition from idle to active
					backwardIdle.append(backwardIdleWindow)
					backwardIdleWindow = 0
			else:
				# Idle on this reading
				backwardIdleWindow += duration
				if backwardActiveWindow > 0:
					# Transition from active to idle
					backwardActive.append(backwardActiveWindow)
					backwardActiveWindow = 0
		# Finish Window if any is open
		if backwardActiveWindow > 0:
			backwardActive.append(backwardActiveWindow)
		if backwardIdleWindow > 0:
			backwardIdle.append(backwardIdleWindow)
		# Add value if no window came to be
		if len(backwardIdle) == 0:
			backwardIdle.append(0)
		if len(backwardActive) == 0:
			backwardActive.append(0)
		# FIAT Details
		fiat_min = min(fiat)
		fiat_max = max(fiat)
		fiat_mean = mean(fiat)
		fiat_std = std(fiat)
		# BIAT Details
		biat_min = min(biat)
		biat_max = max(biat)
		biat_mean = mean(biat)
		biat_std = std(biat)
		# Flow Inter Arrival time
		flowiat = fiat + biat
		flowiat_min = min(flowiat)
		flowiat_max = max(flowiat)
		flowiat_mean = mean(flowiat)
		flowiat_std = std(flowiat)
		# Idle
		idles = backwardIdle + forwardIdle
		idle_min = min(idles)
		idle_max = max(idles)
		idle_mean = mean(idles)
		idle_std = std(idles)
		# Active
		actives = backwardActive + forwardActive
		active_min = min(actives)
		active_max = max(actives)
		active_mean = mean(actives)
		active_std = std(actives)
		print("Duration: {}".format(fullDuration))
		print("fb_psec: {}".format(fb_psec))
		print("fp_psec: {}".format(fp_psec))
		print("fiat_min: {}".format(fiat_min))
		print("fiat_max: {}".format(fiat_max))
		print("fiat_mean: {}".format(fiat_mean))
		print("fiat_std: {}".format(fiat_std))
		print("biat_min: {}".format(biat_min))
		print("biat_max: {}".format(biat_max))
		print("biat_mean: {}".format(biat_mean))
		print("biat_std: {}".format(biat_std))
		print("flowiat_min: {}".format(flowiat_min))
		print("flowiat_max: {}".format(flowiat_max))
		print("flowiat_mean: {}".format(flowiat_mean))
		print("flowiat_std: {}".format(flowiat_std))
		print("idle_min: {}".format(idle_min))
		print("idle_max: {}".format(idle_max))
		print("idle_mean: {}".format(idle_mean))
		print("idle_std: {}".format(idle_std))
		print("active_min: {}".format(active_min))
		print("active_max: {}".format(active_max))
		print("active_mean: {}".format(active_mean))
		print("active_std: {}".format(active_std))
	
	def __eq__(self, other):
		return self.forward == other.forward

class tableStats(EventMixin):
	def __init__(self, interval=5):
		self.fullFlows = []
		self.flows = []
		self.tableActiveCount = {}
		self.interval = interval
		core.openflow.addListeners(self)

	def _handle_ConnectionUp(self, event):
		print("Switch %s has connected" % event.dpid)
		self.sendTableStatsRequest(event)

	def _handle_FlowStatsReceived(self, event):
		print("FlowStatsReceived")
		for f in event.stats:

			#print(f.__dict__)
			#print(f.actions[0].__dict__)
			#print(f.match)

			flow = Flow(f.match.nw_src, f.match.nw_dst, f.match.tp_src if hasattr(f.match, 'tp_src') else None, f.match.tp_dst if hasattr(f.match, 'tp_dst') else None)
			if flow in self.flows:
				# Flow exists: Assign flow to the flow variable
				index = self.flows.index(flow)
				flow = self.flows[index]
			else:
				# Flow doesn't exists: Add flow to list
				self.flows.append(flow)
				# Try to find the Full Flow:
				fullFlow = FullFlow(Flow(flow.dst, flow.src, flow.tp_dst, flow.tp_src))
				if fullFlow in self.fullFlows:
					index = self.fullFlows.index(fullFlow)
					fullFlow = self.fullFlows[index]
					fullFlow.addBackward(flow)
				else:
					self.fullFlows.append(FullFlow(flow))
			# Add reading to flow
			flow.add(f.duration_sec, f.byte_count, f.packet_count)
		for f in self.fullFlows:
			f.printValues()
			if f.isFlow():
				f.printStats()
		Timer(self.interval, self.sendTableStatsRequest, args=[event])

	def sendTableStatsRequest(self, event):
		event.connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
		print("-------------------------------------")
		print("Send flow stat message to Switch %s " % event.dpid)


def launch(interval='5'):
	interval = int(interval)
	core.registerNew(tableStats, interval)
