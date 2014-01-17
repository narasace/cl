#!/usr/bin/env python

import sys; from struct import unpack

file, var = sys.argv[1:]

data = open(file, 'r').read(); l = len(data)

print "extern const unsigned  int %s_len;" % var
print "extern const unsigned char %s_src[];\n" % var

print "const unsigned  int %s_len = %i;" % (var, l)
print "const unsigned char %s_src[] = {" % var,

for c in range(0,l-1):
	if (c % 16 == 0): print "\n\t",
	print "0x%02x," % unpack('B', data[c]),

print "0x00\n};"