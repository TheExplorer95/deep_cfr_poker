??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
??*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
??*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:?*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
??*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
??*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:?*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*'
shared_nameembedding_5/embeddings
?
*embedding_5/embeddings/Read/ReadVariableOpReadVariableOpembedding_5/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameembedding_3/embeddings
?
*embedding_3/embeddings/Read/ReadVariableOpReadVariableOpembedding_3/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameembedding_4/embeddings
?
*embedding_4/embeddings/Read/ReadVariableOpReadVariableOpembedding_4/embeddings*
_output_shapes
:	?*
dtype0
?
normalize/normalization/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namenormalize/normalization/mean
?
0normalize/normalization/mean/Read/ReadVariableOpReadVariableOpnormalize/normalization/mean*
_output_shapes	
:?*
dtype0
?
 normalize/normalization/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" normalize/normalization/variance
?
4normalize/normalization/variance/Read/ReadVariableOpReadVariableOp normalize/normalization/variance*
_output_shapes	
:?*
dtype0
?
normalize/normalization/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *.
shared_namenormalize/normalization/count
?
1normalize/normalization/count/Read/ReadVariableOpReadVariableOpnormalize/normalization/count*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 * $tI
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
?`
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*?`
value?`B?` B?`
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
 
 
?
 layer-0
!layer-1
"layer-2
#layer-3
$layer_with_weights-0
$layer-4
%layer_with_weights-1
%layer-5
&layer-6
'layer-7
(layer-8
)layer_with_weights-2
)layer-9
*layer-10
+layer-11
,layer-12
-layer-13
.layer-14
/	variables
0regularization_losses
1trainable_variables
2	keras_api
?
3layer-0
4layer-1
5layer-2
6layer-3
7layer_with_weights-0
7layer-4
8layer_with_weights-1
8layer-5
9layer-6
:layer-7
;layer-8
<layer_with_weights-2
<layer-9
=layer-10
>layer-11
?layer-12
@layer-13
Alayer-14
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api

F	keras_api

G	keras_api

H	keras_api

I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api

P	keras_api
h

Qkernel
Rbias
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
h

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
h

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
h

ckernel
dbias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api

i	keras_api
h

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api

p	keras_api
h

qkernel
rbias
s	variables
tregularization_losses
utrainable_variables
v	keras_api

w	keras_api

x	keras_api
h

ykernel
zbias
{	variables
|regularization_losses
}trainable_variables
~	keras_api

	keras_api

?	keras_api
f
?	normalize
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
?0
?1
?2
?3
?4
?5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
?22
?23
?24
?25
?26
 
?
?0
?1
?2
?3
?4
?5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
?22
?23
?
 ?layer_regularization_losses
?layers
	variables
?metrics
regularization_losses
?layer_metrics
?non_trainable_variables
trainable_variables
 
 
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2
 

?0
?1
?2
?
 ?layer_regularization_losses
?layers
/	variables
?metrics
0regularization_losses
?layer_metrics
?non_trainable_variables
1trainable_variables
 
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2
 

?0
?1
?2
?
 ?layer_regularization_losses
?layers
B	variables
?metrics
Cregularization_losses
?layer_metrics
?non_trainable_variables
Dtrainable_variables
 
 
 
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
?
 ?layer_regularization_losses
?layers
L	variables
?metrics
Mregularization_losses
?layer_metrics
?non_trainable_variables
Ntrainable_variables
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
?
 ?layer_regularization_losses
?layers
S	variables
?metrics
Tregularization_losses
?layer_metrics
?non_trainable_variables
Utrainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
?
 ?layer_regularization_losses
?layers
Y	variables
?metrics
Zregularization_losses
?layer_metrics
?non_trainable_variables
[trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1
 

]0
^1
?
 ?layer_regularization_losses
?layers
_	variables
?metrics
`regularization_losses
?layer_metrics
?non_trainable_variables
atrainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1
 

c0
d1
?
 ?layer_regularization_losses
?layers
e	variables
?metrics
fregularization_losses
?layer_metrics
?non_trainable_variables
gtrainable_variables
 
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
?
 ?layer_regularization_losses
?layers
l	variables
?metrics
mregularization_losses
?layer_metrics
?non_trainable_variables
ntrainable_variables
 
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
 

q0
r1
?
 ?layer_regularization_losses
?layers
s	variables
?metrics
tregularization_losses
?layer_metrics
?non_trainable_variables
utrainable_variables
 
 
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
 

y0
z1
?
 ?layer_regularization_losses
?layers
{	variables
?metrics
|regularization_losses
?layer_metrics
?non_trainable_variables
}trainable_variables
 
 
c
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api

?0
?1
?2
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
[Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_8/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
RP
VARIABLE_VALUEembedding_2/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEembedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEembedding_1/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEembedding_5/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEembedding_3/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEembedding_4/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEnormalize/normalization/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize/normalization/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnormalize/normalization/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25

?0
 

?0
?1
?2
 
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 
 
 
 
n
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
 
 
 
 
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 
 
 
 
n
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
&
	?mean
?variance

?count
 
 
 

?0
 
 

?0
?1
?2
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
w
serving_default_betsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

y
serving_default_cards0Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_cards1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_2/embeddingsembedding/embeddingsembedding_1/embeddingsConst_2embedding_5/embeddingsembedding_3/embeddingsembedding_4/embeddingsConst_3Const_4dense/kernel
dense/biasdense_3/kerneldense_3/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasnormalize/normalization/mean normalize/normalization/variancedense_8/kerneldense_8/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference_signature_wrapper_2739
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp*embedding_2/embeddings/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp*embedding_5/embeddings/Read/ReadVariableOp*embedding_3/embeddings/Read/ReadVariableOp*embedding_4/embeddings/Read/ReadVariableOp0normalize/normalization/mean/Read/ReadVariableOp4normalize/normalization/variance/Read/ReadVariableOp1normalize/normalization/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst_5**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *&
f!R
__inference__traced_save_3878
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/biasdense_2/kerneldense_2/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasembedding_2/embeddingsembedding/embeddingsembedding_1/embeddingsembedding_5/embeddingsembedding_3/embeddingsembedding_4/embeddingsnormalize/normalization/mean normalize/normalization/variancenormalize/normalization/counttotalcount*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *)
f$R"
 __inference__traced_restore_3975??
?
?
&__inference_model_1_layer_call_fn_1888
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: 
?
p
*__inference_embedding_2_layer_call_fn_3665

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?w
?
 __inference__traced_restore_3975
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_3_kernel#
assignvariableop_5_dense_3_bias%
!assignvariableop_6_dense_2_kernel#
assignvariableop_7_dense_2_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias&
"assignvariableop_10_dense_5_kernel$
 assignvariableop_11_dense_5_bias&
"assignvariableop_12_dense_6_kernel$
 assignvariableop_13_dense_6_bias&
"assignvariableop_14_dense_7_kernel$
 assignvariableop_15_dense_7_bias&
"assignvariableop_16_dense_8_kernel$
 assignvariableop_17_dense_8_bias.
*assignvariableop_18_embedding_2_embeddings,
(assignvariableop_19_embedding_embeddings.
*assignvariableop_20_embedding_1_embeddings.
*assignvariableop_21_embedding_5_embeddings.
*assignvariableop_22_embedding_3_embeddings.
*assignvariableop_23_embedding_4_embeddings4
0assignvariableop_24_normalize_normalization_mean8
4assignvariableop_25_normalize_normalization_variance5
1assignvariableop_26_normalize_normalization_count
assignvariableop_27_total
assignvariableop_28_count
identity_30??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_8_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp*assignvariableop_18_embedding_2_embeddingsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_embedding_embeddingsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_embedding_1_embeddingsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_embedding_5_embeddingsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_embedding_3_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_embedding_4_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_normalize_normalization_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp4assignvariableop_25_normalize_normalization_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp1assignvariableop_26_normalize_normalization_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29?
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*?
_input_shapesx
v: :::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?-
?
A__inference_model_1_layer_call_and_return_conditional_losses_1797
input_2*
&tf_math_greater_equal_1_greaterequal_y
embedding_5_1779
embedding_3_1782
embedding_4_1787
identity??#embedding_3/StatefulPartitionedCall?#embedding_4/StatefulPartitionedCall?#embedding_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_16722
flatten_1/PartitionedCall?
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_1/clip_by_value/Minimum/y?
(tf.clip_by_value_1/clip_by_value/MinimumMinimum"flatten_1/PartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_1/clip_by_value/Minimum?
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_1/clip_by_value/y?
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_1/clip_by_value?
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_1/FloorDiv/y?
!tf.compat.v1.floor_div_1/FloorDivFloorDiv$tf.clip_by_value_1/clip_by_value:z:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_1/FloorDiv?
$tf.math.greater_equal_1/GreaterEqualGreaterEqual"flatten_1/PartitionedCall:output:0&tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_1/GreaterEqual?
tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_1/FloorMod/y?
tf.math.floormod_1/FloorModFloorMod$tf.clip_by_value_1/clip_by_value:z:0&tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_1/FloorMod?
#embedding_5/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_1/clip_by_value:z:0embedding_5_1779*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_5_layer_call_and_return_conditional_losses_17002%
#embedding_5/StatefulPartitionedCall?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_1/FloorDiv:z:0embedding_3_1782*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_17222%
#embedding_3/StatefulPartitionedCall?
tf.cast_1/CastCast(tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_1/Cast?
tf.__operators__.add_2/AddV2AddV2,embedding_5/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_2/AddV2?
#embedding_4/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_1/FloorMod:z:0embedding_4_1787*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_17462%
#embedding_4/StatefulPartitionedCall?
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0,embedding_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_3/AddV2?
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_1/ExpandDims/dim?
tf.expand_dims_1/ExpandDims
ExpandDimstf.cast_1/Cast:y:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_1/ExpandDims?
tf.math.multiply_1/MulMul tf.__operators__.add_3/AddV2:z:0$tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_1/Mul?
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_1/Sum/reduction_indices?
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_1/Sum?
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: 
?
?
&__inference_model_1_layer_call_fn_1843
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: 
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_3675

inputs
embedding_lookup_3669
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3669Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/3669*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3669*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
?__inference_model_layer_call_and_return_conditional_losses_1651

inputs(
$tf_math_greater_equal_greaterequal_y
embedding_2_1633
embedding_1636
embedding_1_1641
identity??!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14462
flatten/PartitionedCall?
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2*
(tf.clip_by_value/clip_by_value/Minimum/y?
&tf.clip_by_value/clip_by_value/MinimumMinimum flatten/PartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/y?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!tf.compat.v1.floor_div/FloorDiv/y?
tf.compat.v1.floor_div/FloorDivFloorDiv"tf.clip_by_value/clip_by_value:z:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2!
tf.compat.v1.floor_div/FloorDiv?
"tf.math.greater_equal/GreaterEqualGreaterEqual flatten/PartitionedCall:output:0$tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2$
"tf.math.greater_equal/GreaterEqual
tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod/FloorMod/y?
tf.math.floormod/FloorModFloorMod"tf.clip_by_value/clip_by_value:z:0$tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod/FloorMod?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"tf.clip_by_value/clip_by_value:z:0embedding_2_1633*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14742%
#embedding_2/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCall#tf.compat.v1.floor_div/FloorDiv:z:0embedding_1636*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_14962#
!embedding/StatefulPartitionedCall?
tf.cast/CastCast&tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast/Cast?
tf.__operators__.add/AddV2AddV2,embedding_2/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add/AddV2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod/FloorMod:z:0embedding_1_1641*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15202%
#embedding_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0,embedding_1/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_1/AddV2?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimstf.cast/Cast:y:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims/ExpandDims?
tf.math.multiply/MulMul tf.__operators__.add_1/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply/Mul?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum/Sum?
IdentityIdentitytf.math.reduce_sum/Sum:output:0"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
p
*__inference_embedding_1_layer_call_fn_3699

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
(__inference_embedding_layer_call_fn_3682

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_14962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?T
?	
F__inference_custom_model_layer_call_and_return_conditional_losses_2442

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_2_greaterequal_y

model_2357

model_2359

model_2361

model_2363
model_1_2366
model_1_2368
model_1_2370
model_1_2372.
*tf_clip_by_value_2_clip_by_value_minimum_y&
"tf_clip_by_value_2_clip_by_value_y

dense_2384

dense_2386
dense_3_2389
dense_3_2391
dense_1_2394
dense_1_2396
dense_2_2399
dense_2_2401
dense_4_2404
dense_4_2406
dense_5_2411
dense_5_2413
dense_6_2417
dense_6_2419
dense_7_2424
dense_7_2426
normalize_2431
normalize_2433
dense_8_2436
dense_8_2438
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?!normalize/StatefulPartitionedCall?
$tf.math.greater_equal_2/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_2/GreaterEqual?
model/StatefulPartitionedCallStatefulPartitionedCallinputs
model_2357
model_2359
model_2361
model_2363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16062
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_1_2366model_1_2368model_1_2370model_1_2372*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18322!
model_1/StatefulPartitionedCall?
(tf.clip_by_value_2/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0"tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_2/clip_by_value?
tf.cast_2/CastCast(tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_2/Castp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat/concat}
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2$tf.clip_by_value_2/clip_by_value:z:0tf.cast_2/Cast:y:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_1/concat?
dense/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0
dense_2384
dense_2386*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19862
dense/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dense_3_2389dense_3_2391*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_20122!
dense_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2394dense_1_2396*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_20392!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2399dense_2_2401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_20662!
dense_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_2404dense_4_2406*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_20922!
dense_4/StatefulPartitionedCall}
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
dense_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dense_5_2411dense_5_2413*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_21202!
dense_5/StatefulPartitionedCall?
tf.nn.relu/ReluRelu(dense_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu/Relu?
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu/Relu:activations:0dense_6_2417dense_6_2419*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_21472!
dense_6/StatefulPartitionedCall?
tf.__operators__.add_4/AddV2AddV2(dense_6/StatefulPartitionedCall:output:0tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_4/AddV2?
tf.nn.relu_1/ReluRelu tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_1/Relu?
dense_7/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_1/Relu:activations:0dense_7_2424dense_7_2426*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_21752!
dense_7/StatefulPartitionedCall?
tf.__operators__.add_5/AddV2AddV2(dense_7/StatefulPartitionedCall:output:0tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_5/AddV2?
tf.nn.relu_2/ReluRelu tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_2/Relu?
!normalize/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_2/Relu:activations:0normalize_2431normalize_2433*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_normalize_layer_call_and_return_conditional_losses_22102#
!normalize/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*normalize/StatefulPartitionedCall:output:0dense_8_2436dense_8_2438*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_22362!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^normalize/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!normalize/StatefulPartitionedCall!normalize/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?-
?
A__inference_model_1_layer_call_and_return_conditional_losses_1765
input_2*
&tf_math_greater_equal_1_greaterequal_y
embedding_5_1709
embedding_3_1731
embedding_4_1755
identity??#embedding_3/StatefulPartitionedCall?#embedding_4/StatefulPartitionedCall?#embedding_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_16722
flatten_1/PartitionedCall?
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_1/clip_by_value/Minimum/y?
(tf.clip_by_value_1/clip_by_value/MinimumMinimum"flatten_1/PartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_1/clip_by_value/Minimum?
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_1/clip_by_value/y?
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_1/clip_by_value?
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_1/FloorDiv/y?
!tf.compat.v1.floor_div_1/FloorDivFloorDiv$tf.clip_by_value_1/clip_by_value:z:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_1/FloorDiv?
$tf.math.greater_equal_1/GreaterEqualGreaterEqual"flatten_1/PartitionedCall:output:0&tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_1/GreaterEqual?
tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_1/FloorMod/y?
tf.math.floormod_1/FloorModFloorMod$tf.clip_by_value_1/clip_by_value:z:0&tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_1/FloorMod?
#embedding_5/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_1/clip_by_value:z:0embedding_5_1709*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_5_layer_call_and_return_conditional_losses_17002%
#embedding_5/StatefulPartitionedCall?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_1/FloorDiv:z:0embedding_3_1731*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_17222%
#embedding_3/StatefulPartitionedCall?
tf.cast_1/CastCast(tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_1/Cast?
tf.__operators__.add_2/AddV2AddV2,embedding_5/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_2/AddV2?
#embedding_4/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_1/FloorMod:z:0embedding_4_1755*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_17462%
#embedding_4/StatefulPartitionedCall?
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0,embedding_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_3/AddV2?
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_1/ExpandDims/dim?
tf.expand_dims_1/ExpandDims
ExpandDimstf.cast_1/Cast:y:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_1/ExpandDims?
tf.math.multiply_1/MulMul tf.__operators__.add_3/AddV2:z:0$tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_1/Mul?
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_1/Sum/reduction_indices?
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_1/Sum?
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: 
?
p
*__inference_embedding_3_layer_call_fn_3744

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_17222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_1662
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
p
*__inference_embedding_5_layer_call_fn_3727

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_5_layer_call_and_return_conditional_losses_17002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_1_layer_call_and_return_conditional_losses_1520

inputs
embedding_lookup_1514
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_1514Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/1514*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1514*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_1446

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_8_layer_call_and_return_conditional_losses_3628

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_3327

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?6
?
?__inference_model_layer_call_and_return_conditional_losses_3259

inputs(
$tf_math_greater_equal_greaterequal_y%
!embedding_2_embedding_lookup_3233#
embedding_embedding_lookup_3239%
!embedding_1_embedding_lookup_3247
identity??embedding/embedding_lookup?embedding_1/embedding_lookup?embedding_2/embedding_lookupo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten/Reshape?
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2*
(tf.clip_by_value/clip_by_value/Minimum/y?
&tf.clip_by_value/clip_by_value/MinimumMinimumflatten/Reshape:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/y?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!tf.compat.v1.floor_div/FloorDiv/y?
tf.compat.v1.floor_div/FloorDivFloorDiv"tf.clip_by_value/clip_by_value:z:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2!
tf.compat.v1.floor_div/FloorDiv?
"tf.math.greater_equal/GreaterEqualGreaterEqualflatten/Reshape:output:0$tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2$
"tf.math.greater_equal/GreaterEqual
tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod/FloorMod/y?
tf.math.floormod/FloorModFloorMod"tf.clip_by_value/clip_by_value:z:0$tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod/FloorMod?
embedding_2/CastCast"tf.clip_by_value/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather!embedding_2_embedding_lookup_3233embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_2/embedding_lookup/3233*,
_output_shapes
:??????????*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_2/embedding_lookup/3233*,
_output_shapes
:??????????2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_2/embedding_lookup/Identity_1?
embedding/CastCast#tf.compat.v1.floor_div/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_3239embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/3239*,
_output_shapes
:??????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/3239*,
_output_shapes
:??????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2'
%embedding/embedding_lookup/Identity_1?
tf.cast/CastCast&tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast/Cast?
tf.__operators__.add/AddV2AddV20embedding_2/embedding_lookup/Identity_1:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add/AddV2?
embedding_1/CastCasttf.math.floormod/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_3247embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/3247*,
_output_shapes
:??????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/3247*,
_output_shapes
:??????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_1/embedding_lookup/Identity_1?
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_1/AddV2?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimstf.cast/Cast:y:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims/ExpandDims?
tf.math.multiply/MulMul tf.__operators__.add_1/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply/Mul?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum/Sum?
IdentityIdentitytf.math.reduce_sum/Sum:output:0^embedding/embedding_lookup^embedding_1/embedding_lookup^embedding_2/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_3643

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_1496

inputs
embedding_lookup_1490
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_1490Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/1490*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1490*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?T
?	
F__inference_custom_model_layer_call_and_return_conditional_losses_2345

cards0

cards1
bets*
&tf_math_greater_equal_2_greaterequal_y

model_2260

model_2262

model_2264

model_2266
model_1_2269
model_1_2271
model_1_2273
model_1_2275.
*tf_clip_by_value_2_clip_by_value_minimum_y&
"tf_clip_by_value_2_clip_by_value_y

dense_2287

dense_2289
dense_3_2292
dense_3_2294
dense_1_2297
dense_1_2299
dense_2_2302
dense_2_2304
dense_4_2307
dense_4_2309
dense_5_2314
dense_5_2316
dense_6_2320
dense_6_2322
dense_7_2327
dense_7_2329
normalize_2334
normalize_2336
dense_8_2339
dense_8_2341
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?!normalize/StatefulPartitionedCall?
$tf.math.greater_equal_2/GreaterEqualGreaterEqualbets&tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_2/GreaterEqual?
model/StatefulPartitionedCallStatefulPartitionedCallcards0
model_2260
model_2262
model_2264
model_2266*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16512
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallcards1model_1_2269model_1_2271model_1_2273model_1_2275*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18772!
model_1/StatefulPartitionedCall?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumbets*tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0"tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_2/clip_by_value?
tf.cast_2/CastCast(tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_2/Castp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat/concat}
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2$tf.clip_by_value_2/clip_by_value:z:0tf.cast_2/Cast:y:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_1/concat?
dense/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0
dense_2287
dense_2289*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19862
dense/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dense_3_2292dense_3_2294*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_20122!
dense_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2297dense_1_2299*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_20392!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2302dense_2_2304*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_20662!
dense_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_2307dense_4_2309*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_20922!
dense_4/StatefulPartitionedCall}
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
dense_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dense_5_2314dense_5_2316*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_21202!
dense_5/StatefulPartitionedCall?
tf.nn.relu/ReluRelu(dense_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu/Relu?
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu/Relu:activations:0dense_6_2320dense_6_2322*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_21472!
dense_6/StatefulPartitionedCall?
tf.__operators__.add_4/AddV2AddV2(dense_6/StatefulPartitionedCall:output:0tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_4/AddV2?
tf.nn.relu_1/ReluRelu tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_1/Relu?
dense_7/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_1/Relu:activations:0dense_7_2327dense_7_2329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_21752!
dense_7/StatefulPartitionedCall?
tf.__operators__.add_5/AddV2AddV2(dense_7/StatefulPartitionedCall:output:0tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_5/AddV2?
tf.nn.relu_2/ReluRelu tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_2/Relu?
!normalize/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_2/Relu:activations:0normalize_2334normalize_2336*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_normalize_layer_call_and_return_conditional_losses_22102#
!normalize/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*normalize/StatefulPartitionedCall:output:0dense_8_2339dense_8_2341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_22362!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^normalize/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!normalize/StatefulPartitionedCall!normalize/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?T
?	
F__inference_custom_model_layer_call_and_return_conditional_losses_2603

inputs
inputs_1
inputs_2*
&tf_math_greater_equal_2_greaterequal_y

model_2518

model_2520

model_2522

model_2524
model_1_2527
model_1_2529
model_1_2531
model_1_2533.
*tf_clip_by_value_2_clip_by_value_minimum_y&
"tf_clip_by_value_2_clip_by_value_y

dense_2545

dense_2547
dense_3_2550
dense_3_2552
dense_1_2555
dense_1_2557
dense_2_2560
dense_2_2562
dense_4_2565
dense_4_2567
dense_5_2572
dense_5_2574
dense_6_2578
dense_6_2580
dense_7_2585
dense_7_2587
normalize_2592
normalize_2594
dense_8_2597
dense_8_2599
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?!normalize/StatefulPartitionedCall?
$tf.math.greater_equal_2/GreaterEqualGreaterEqualinputs_2&tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_2/GreaterEqual?
model/StatefulPartitionedCallStatefulPartitionedCallinputs
model_2518
model_2520
model_2522
model_2524*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16512
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_1_2527model_1_2529model_1_2531model_1_2533*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18772!
model_1/StatefulPartitionedCall?
(tf.clip_by_value_2/clip_by_value/MinimumMinimuminputs_2*tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0"tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_2/clip_by_value?
tf.cast_2/CastCast(tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_2/Castp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat/concat}
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2$tf.clip_by_value_2/clip_by_value:z:0tf.cast_2/Cast:y:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_1/concat?
dense/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0
dense_2545
dense_2547*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19862
dense/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dense_3_2550dense_3_2552*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_20122!
dense_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2555dense_1_2557*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_20392!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2560dense_2_2562*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_20662!
dense_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_2565dense_4_2567*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_20922!
dense_4/StatefulPartitionedCall}
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
dense_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dense_5_2572dense_5_2574*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_21202!
dense_5/StatefulPartitionedCall?
tf.nn.relu/ReluRelu(dense_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu/Relu?
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu/Relu:activations:0dense_6_2578dense_6_2580*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_21472!
dense_6/StatefulPartitionedCall?
tf.__operators__.add_4/AddV2AddV2(dense_6/StatefulPartitionedCall:output:0tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_4/AddV2?
tf.nn.relu_1/ReluRelu tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_1/Relu?
dense_7/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_1/Relu:activations:0dense_7_2585dense_7_2587*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_21752!
dense_7/StatefulPartitionedCall?
tf.__operators__.add_5/AddV2AddV2(dense_7/StatefulPartitionedCall:output:0tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_5/AddV2?
tf.nn.relu_2/ReluRelu tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_2/Relu?
!normalize/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_2/Relu:activations:0normalize_2592normalize_2594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_normalize_layer_call_and_return_conditional_losses_22102#
!normalize/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*normalize/StatefulPartitionedCall:output:0dense_8_2597dense_8_2599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_22362!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^normalize/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!normalize/StatefulPartitionedCall!normalize/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?T
?	
F__inference_custom_model_layer_call_and_return_conditional_losses_2253

cards0

cards1
bets*
&tf_math_greater_equal_2_greaterequal_y

model_1922

model_1924

model_1926

model_1928
model_1_1957
model_1_1959
model_1_1961
model_1_1963.
*tf_clip_by_value_2_clip_by_value_minimum_y&
"tf_clip_by_value_2_clip_by_value_y

dense_1997

dense_1999
dense_3_2023
dense_3_2025
dense_1_2050
dense_1_2052
dense_2_2077
dense_2_2079
dense_4_2103
dense_4_2105
dense_5_2131
dense_5_2133
dense_6_2158
dense_6_2160
dense_7_2186
dense_7_2188
normalize_2221
normalize_2223
dense_8_2247
dense_8_2249
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?!normalize/StatefulPartitionedCall?
$tf.math.greater_equal_2/GreaterEqualGreaterEqualbets&tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_2/GreaterEqual?
model/StatefulPartitionedCallStatefulPartitionedCallcards0
model_1922
model_1924
model_1926
model_1928*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16062
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallcards1model_1_1957model_1_1959model_1_1961model_1_1963*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18322!
model_1/StatefulPartitionedCall?
(tf.clip_by_value_2/clip_by_value/MinimumMinimumbets*tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0"tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_2/clip_by_value?
tf.cast_2/CastCast(tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_2/Castp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat/concat}
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2$tf.clip_by_value_2/clip_by_value:z:0tf.cast_2/Cast:y:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_1/concat?
dense/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0
dense_1997
dense_1999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19862
dense/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0dense_3_2023dense_3_2025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_20122!
dense_3/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2050dense_1_2052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_20392!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2077dense_2_2079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_20662!
dense_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_2103dense_4_2105*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_20922!
dense_4/StatefulPartitionedCall}
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2(dense_2/StatefulPartitionedCall:output:0(dense_4/StatefulPartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
dense_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0dense_5_2131dense_5_2133*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_21202!
dense_5/StatefulPartitionedCall?
tf.nn.relu/ReluRelu(dense_5/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu/Relu?
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu/Relu:activations:0dense_6_2158dense_6_2160*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_21472!
dense_6/StatefulPartitionedCall?
tf.__operators__.add_4/AddV2AddV2(dense_6/StatefulPartitionedCall:output:0tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_4/AddV2?
tf.nn.relu_1/ReluRelu tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_1/Relu?
dense_7/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_1/Relu:activations:0dense_7_2186dense_7_2188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_21752!
dense_7/StatefulPartitionedCall?
tf.__operators__.add_5/AddV2AddV2(dense_7/StatefulPartitionedCall:output:0tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_5/AddV2?
tf.nn.relu_2/ReluRelu tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_2/Relu?
!normalize/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_2/Relu:activations:0normalize_2221normalize_2223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_normalize_layer_call_and_return_conditional_losses_22102#
!normalize/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall*normalize/StatefulPartitionedCall:output:0dense_8_2247dense_8_2249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_22362!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall"^normalize/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall2F
!normalize/StatefulPartitionedCall!normalize/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_dense_5_layer_call_and_return_conditional_losses_2120

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_5_layer_call_and_return_conditional_losses_3720

inputs
embedding_lookup_3714
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3714Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/3714*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3714*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
__inference__traced_save_3878
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop5
1savev2_embedding_2_embeddings_read_readvariableop3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop5
1savev2_embedding_5_embeddings_read_readvariableop5
1savev2_embedding_3_embeddings_read_readvariableop5
1savev2_embedding_4_embeddings_read_readvariableop;
7savev2_normalize_normalization_mean_read_readvariableop?
;savev2_normalize_normalization_variance_read_readvariableop<
8savev2_normalize_normalization_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const_5

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop1savev2_embedding_2_embeddings_read_readvariableop/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop1savev2_embedding_5_embeddings_read_readvariableop1savev2_embedding_3_embeddings_read_readvariableop1savev2_embedding_4_embeddings_read_readvariableop7savev2_normalize_normalization_mean_read_readvariableop;savev2_normalize_normalization_variance_read_readvariableop8savev2_normalize_normalization_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?:?:?: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_dense_8_layer_call_fn_3637

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_8_layer_call_and_return_conditional_losses_22362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_6_layer_call_and_return_conditional_losses_3564

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_dense_6_layer_call_fn_3573

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_21472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_normalize_layer_call_and_return_conditional_losses_3609
x1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization/Reshape_1?
normalization/subSubxnormalization/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization/sub|
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization/truediv?
IdentityIdentitynormalization/truediv:z:0%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
y
$__inference_dense_layer_call_fn_3457

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_19862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_1986

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
F__inference_custom_model_layer_call_and_return_conditional_losses_3079

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_2_greaterequal_y.
*model_tf_math_greater_equal_greaterequal_y+
'model_embedding_2_embedding_lookup_2929)
%model_embedding_embedding_lookup_2935+
'model_embedding_1_embedding_lookup_29432
.model_1_tf_math_greater_equal_1_greaterequal_y-
)model_1_embedding_5_embedding_lookup_2967-
)model_1_embedding_3_embedding_lookup_2973-
)model_1_embedding_4_embedding_lookup_2981.
*tf_clip_by_value_2_clip_by_value_minimum_y&
"tf_clip_by_value_2_clip_by_value_y(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource;
7normalize_normalization_reshape_readvariableop_resource=
9normalize_normalization_reshape_1_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp? model/embedding/embedding_lookup?"model/embedding_1/embedding_lookup?"model/embedding_2/embedding_lookup?$model_1/embedding_3/embedding_lookup?$model_1/embedding_4/embedding_lookup?$model_1/embedding_5/embedding_lookup?.normalize/normalization/Reshape/ReadVariableOp?0normalize/normalization/Reshape_1/ReadVariableOp?
$tf.math.greater_equal_2/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_2/GreaterEqual{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten/Const?
model/flatten/ReshapeReshape
inputs_0_0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten/Reshape?
.model/tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI20
.model/tf.clip_by_value/clip_by_value/Minimum/y?
,model/tf.clip_by_value/clip_by_value/MinimumMinimummodel/flatten/Reshape:output:07model/tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2.
,model/tf.clip_by_value/clip_by_value/Minimum?
&model/tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&model/tf.clip_by_value/clip_by_value/y?
$model/tf.clip_by_value/clip_by_valueMaximum0model/tf.clip_by_value/clip_by_value/Minimum:z:0/model/tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2&
$model/tf.clip_by_value/clip_by_value?
'model/tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model/tf.compat.v1.floor_div/FloorDiv/y?
%model/tf.compat.v1.floor_div/FloorDivFloorDiv(model/tf.clip_by_value/clip_by_value:z:00model/tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2'
%model/tf.compat.v1.floor_div/FloorDiv?
(model/tf.math.greater_equal/GreaterEqualGreaterEqualmodel/flatten/Reshape:output:0*model_tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2*
(model/tf.math.greater_equal/GreaterEqual?
!model/tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!model/tf.math.floormod/FloorMod/y?
model/tf.math.floormod/FloorModFloorMod(model/tf.clip_by_value/clip_by_value:z:0*model/tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2!
model/tf.math.floormod/FloorMod?
model/embedding_2/CastCast(model/tf.clip_by_value/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding_2/Cast?
"model/embedding_2/embedding_lookupResourceGather'model_embedding_2_embedding_lookup_2929model/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@model/embedding_2/embedding_lookup/2929*,
_output_shapes
:??????????*
dtype02$
"model/embedding_2/embedding_lookup?
+model/embedding_2/embedding_lookup/IdentityIdentity+model/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@model/embedding_2/embedding_lookup/2929*,
_output_shapes
:??????????2-
+model/embedding_2/embedding_lookup/Identity?
-model/embedding_2/embedding_lookup/Identity_1Identity4model/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2/
-model/embedding_2/embedding_lookup/Identity_1?
model/embedding/CastCast)model/tf.compat.v1.floor_div/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding/Cast?
 model/embedding/embedding_lookupResourceGather%model_embedding_embedding_lookup_2935model/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@model/embedding/embedding_lookup/2935*,
_output_shapes
:??????????*
dtype02"
 model/embedding/embedding_lookup?
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@model/embedding/embedding_lookup/2935*,
_output_shapes
:??????????2+
)model/embedding/embedding_lookup/Identity?
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2-
+model/embedding/embedding_lookup/Identity_1?
model/tf.cast/CastCast,model/tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model/tf.cast/Cast?
 model/tf.__operators__.add/AddV2AddV26model/embedding_2/embedding_lookup/Identity_1:output:04model/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2"
 model/tf.__operators__.add/AddV2?
model/embedding_1/CastCast#model/tf.math.floormod/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding_1/Cast?
"model/embedding_1/embedding_lookupResourceGather'model_embedding_1_embedding_lookup_2943model/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@model/embedding_1/embedding_lookup/2943*,
_output_shapes
:??????????*
dtype02$
"model/embedding_1/embedding_lookup?
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@model/embedding_1/embedding_lookup/2943*,
_output_shapes
:??????????2-
+model/embedding_1/embedding_lookup/Identity?
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2/
-model/embedding_1/embedding_lookup/Identity_1?
"model/tf.__operators__.add_1/AddV2AddV2$model/tf.__operators__.add/AddV2:z:06model/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2$
"model/tf.__operators__.add_1/AddV2?
#model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model/tf.expand_dims/ExpandDims/dim?
model/tf.expand_dims/ExpandDims
ExpandDimsmodel/tf.cast/Cast:y:0,model/tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2!
model/tf.expand_dims/ExpandDims?
model/tf.math.multiply/MulMul&model/tf.__operators__.add_1/AddV2:z:0(model/tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
model/tf.math.multiply/Mul?
.model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.model/tf.math.reduce_sum/Sum/reduction_indices?
model/tf.math.reduce_sum/SumSummodel/tf.math.multiply/Mul:z:07model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
model/tf.math.reduce_sum/Sum?
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_1/Const?
model_1/flatten_1/ReshapeReshape
inputs_0_1 model_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_1/Reshape?
2model_1/tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_1/tf.clip_by_value_1/clip_by_value/Minimum/y?
0model_1/tf.clip_by_value_1/clip_by_value/MinimumMinimum"model_1/flatten_1/Reshape:output:0;model_1/tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_1/tf.clip_by_value_1/clip_by_value/Minimum?
*model_1/tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_1/tf.clip_by_value_1/clip_by_value/y?
(model_1/tf.clip_by_value_1/clip_by_valueMaximum4model_1/tf.clip_by_value_1/clip_by_value/Minimum:z:03model_1/tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_1/tf.clip_by_value_1/clip_by_value?
+model_1/tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_1/tf.compat.v1.floor_div_1/FloorDiv/y?
)model_1/tf.compat.v1.floor_div_1/FloorDivFloorDiv,model_1/tf.clip_by_value_1/clip_by_value:z:04model_1/tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_1/tf.compat.v1.floor_div_1/FloorDiv?
,model_1/tf.math.greater_equal_1/GreaterEqualGreaterEqual"model_1/flatten_1/Reshape:output:0.model_1_tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_1/tf.math.greater_equal_1/GreaterEqual?
%model_1/tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_1/tf.math.floormod_1/FloorMod/y?
#model_1/tf.math.floormod_1/FloorModFloorMod,model_1/tf.clip_by_value_1/clip_by_value:z:0.model_1/tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_1/tf.math.floormod_1/FloorMod?
model_1/embedding_5/CastCast,model_1/tf.clip_by_value_1/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_5/Cast?
$model_1/embedding_5/embedding_lookupResourceGather)model_1_embedding_5_embedding_lookup_2967model_1/embedding_5/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_1/embedding_5/embedding_lookup/2967*,
_output_shapes
:??????????*
dtype02&
$model_1/embedding_5/embedding_lookup?
-model_1/embedding_5/embedding_lookup/IdentityIdentity-model_1/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_1/embedding_5/embedding_lookup/2967*,
_output_shapes
:??????????2/
-model_1/embedding_5/embedding_lookup/Identity?
/model_1/embedding_5/embedding_lookup/Identity_1Identity6model_1/embedding_5/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????21
/model_1/embedding_5/embedding_lookup/Identity_1?
model_1/embedding_3/CastCast-model_1/tf.compat.v1.floor_div_1/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_3/Cast?
$model_1/embedding_3/embedding_lookupResourceGather)model_1_embedding_3_embedding_lookup_2973model_1/embedding_3/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_1/embedding_3/embedding_lookup/2973*,
_output_shapes
:??????????*
dtype02&
$model_1/embedding_3/embedding_lookup?
-model_1/embedding_3/embedding_lookup/IdentityIdentity-model_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_1/embedding_3/embedding_lookup/2973*,
_output_shapes
:??????????2/
-model_1/embedding_3/embedding_lookup/Identity?
/model_1/embedding_3/embedding_lookup/Identity_1Identity6model_1/embedding_3/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????21
/model_1/embedding_3/embedding_lookup/Identity_1?
model_1/tf.cast_1/CastCast0model_1/tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_1/tf.cast_1/Cast?
$model_1/tf.__operators__.add_2/AddV2AddV28model_1/embedding_5/embedding_lookup/Identity_1:output:08model_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2&
$model_1/tf.__operators__.add_2/AddV2?
model_1/embedding_4/CastCast'model_1/tf.math.floormod_1/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_4/Cast?
$model_1/embedding_4/embedding_lookupResourceGather)model_1_embedding_4_embedding_lookup_2981model_1/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_1/embedding_4/embedding_lookup/2981*,
_output_shapes
:??????????*
dtype02&
$model_1/embedding_4/embedding_lookup?
-model_1/embedding_4/embedding_lookup/IdentityIdentity-model_1/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_1/embedding_4/embedding_lookup/2981*,
_output_shapes
:??????????2/
-model_1/embedding_4/embedding_lookup/Identity?
/model_1/embedding_4/embedding_lookup/Identity_1Identity6model_1/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????21
/model_1/embedding_4/embedding_lookup/Identity_1?
$model_1/tf.__operators__.add_3/AddV2AddV2(model_1/tf.__operators__.add_2/AddV2:z:08model_1/embedding_4/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2&
$model_1/tf.__operators__.add_3/AddV2?
'model_1/tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_1/tf.expand_dims_1/ExpandDims/dim?
#model_1/tf.expand_dims_1/ExpandDims
ExpandDimsmodel_1/tf.cast_1/Cast:y:00model_1/tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_1/tf.expand_dims_1/ExpandDims?
model_1/tf.math.multiply_1/MulMul(model_1/tf.__operators__.add_3/AddV2:z:0,model_1/tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_1/tf.math.multiply_1/Mul?
2model_1/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_1/tf.math.reduce_sum_1/Sum/reduction_indices?
 model_1/tf.math.reduce_sum_1/SumSum"model_1/tf.math.multiply_1/Mul:z:0;model_1/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_1/tf.math.reduce_sum_1/Sum?
(tf.clip_by_value_2/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0"tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_2/clip_by_value?
tf.cast_2/CastCast(tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_2/Castp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2%model/tf.math.reduce_sum/Sum:output:0)model_1/tf.math.reduce_sum_1/Sum:output:0tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat/concat}
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2$tf.clip_by_value_2/clip_by_value:z:0tf.cast_2/Cast:y:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_1/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMultf.concat/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMultf.concat_1/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/BiasAdd:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAdd}
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2dense_2/Relu:activations:0dense_4/BiasAdd:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMultf.concat_2/concat:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddw
tf.nn.relu/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu/Relu?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMultf.nn.relu/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAdd?
tf.__operators__.add_4/AddV2AddV2dense_6/BiasAdd:output:0tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_4/AddV2?
tf.nn.relu_1/ReluRelu tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_1/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMultf.nn.relu_1/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAdd?
tf.__operators__.add_5/AddV2AddV2dense_7/BiasAdd:output:0tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_5/AddV2?
tf.nn.relu_2/ReluRelu tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_2/Relu?
.normalize/normalization/Reshape/ReadVariableOpReadVariableOp7normalize_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype020
.normalize/normalization/Reshape/ReadVariableOp?
%normalize/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%normalize/normalization/Reshape/shape?
normalize/normalization/ReshapeReshape6normalize/normalization/Reshape/ReadVariableOp:value:0.normalize/normalization/Reshape/shape:output:0*
T0*
_output_shapes
:	?2!
normalize/normalization/Reshape?
0normalize/normalization/Reshape_1/ReadVariableOpReadVariableOp9normalize_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype022
0normalize/normalization/Reshape_1/ReadVariableOp?
'normalize/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2)
'normalize/normalization/Reshape_1/shape?
!normalize/normalization/Reshape_1Reshape8normalize/normalization/Reshape_1/ReadVariableOp:value:00normalize/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2#
!normalize/normalization/Reshape_1?
normalize/normalization/subSubtf.nn.relu_2/Relu:activations:0(normalize/normalization/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalize/normalization/sub?
normalize/normalization/SqrtSqrt*normalize/normalization/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalize/normalization/Sqrt?
!normalize/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32#
!normalize/normalization/Maximum/y?
normalize/normalization/MaximumMaximum normalize/normalization/Sqrt:y:0*normalize/normalization/Maximum/y:output:0*
T0*
_output_shapes
:	?2!
normalize/normalization/Maximum?
normalize/normalization/truedivRealDivnormalize/normalization/sub:z:0#normalize/normalization/Maximum:z:0*
T0*(
_output_shapes
:??????????2!
normalize/normalization/truediv?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMul#normalize/normalization/truediv:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdd?
IdentityIdentitydense_8/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp!^model/embedding/embedding_lookup#^model/embedding_1/embedding_lookup#^model/embedding_2/embedding_lookup%^model_1/embedding_3/embedding_lookup%^model_1/embedding_4/embedding_lookup%^model_1/embedding_5/embedding_lookup/^normalize/normalization/Reshape/ReadVariableOp1^normalize/normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2D
 model/embedding/embedding_lookup model/embedding/embedding_lookup2H
"model/embedding_1/embedding_lookup"model/embedding_1/embedding_lookup2H
"model/embedding_2/embedding_lookup"model/embedding_2/embedding_lookup2L
$model_1/embedding_3/embedding_lookup$model_1/embedding_3/embedding_lookup2L
$model_1/embedding_4/embedding_lookup$model_1/embedding_4/embedding_lookup2L
$model_1/embedding_5/embedding_lookup$model_1/embedding_5/embedding_lookup2`
.normalize/normalization/Reshape/ReadVariableOp.normalize/normalization/Reshape/ReadVariableOp2d
0normalize/normalization/Reshape_1/ReadVariableOp0normalize/normalization/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_2039

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
F__inference_custom_model_layer_call_and_return_conditional_losses_2909

inputs_0_0

inputs_0_1
inputs_1*
&tf_math_greater_equal_2_greaterequal_y.
*model_tf_math_greater_equal_greaterequal_y+
'model_embedding_2_embedding_lookup_2759)
%model_embedding_embedding_lookup_2765+
'model_embedding_1_embedding_lookup_27732
.model_1_tf_math_greater_equal_1_greaterequal_y-
)model_1_embedding_5_embedding_lookup_2797-
)model_1_embedding_3_embedding_lookup_2803-
)model_1_embedding_4_embedding_lookup_2811.
*tf_clip_by_value_2_clip_by_value_minimum_y&
"tf_clip_by_value_2_clip_by_value_y(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource;
7normalize_normalization_reshape_readvariableop_resource=
9normalize_normalization_reshape_1_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp? model/embedding/embedding_lookup?"model/embedding_1/embedding_lookup?"model/embedding_2/embedding_lookup?$model_1/embedding_3/embedding_lookup?$model_1/embedding_4/embedding_lookup?$model_1/embedding_5/embedding_lookup?.normalize/normalization/Reshape/ReadVariableOp?0normalize/normalization/Reshape_1/ReadVariableOp?
$tf.math.greater_equal_2/GreaterEqualGreaterEqualinputs_1&tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
2&
$tf.math.greater_equal_2/GreaterEqual{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatten/Const?
model/flatten/ReshapeReshape
inputs_0_0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
model/flatten/Reshape?
.model/tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI20
.model/tf.clip_by_value/clip_by_value/Minimum/y?
,model/tf.clip_by_value/clip_by_value/MinimumMinimummodel/flatten/Reshape:output:07model/tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2.
,model/tf.clip_by_value/clip_by_value/Minimum?
&model/tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&model/tf.clip_by_value/clip_by_value/y?
$model/tf.clip_by_value/clip_by_valueMaximum0model/tf.clip_by_value/clip_by_value/Minimum:z:0/model/tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2&
$model/tf.clip_by_value/clip_by_value?
'model/tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2)
'model/tf.compat.v1.floor_div/FloorDiv/y?
%model/tf.compat.v1.floor_div/FloorDivFloorDiv(model/tf.clip_by_value/clip_by_value:z:00model/tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2'
%model/tf.compat.v1.floor_div/FloorDiv?
(model/tf.math.greater_equal/GreaterEqualGreaterEqualmodel/flatten/Reshape:output:0*model_tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2*
(model/tf.math.greater_equal/GreaterEqual?
!model/tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!model/tf.math.floormod/FloorMod/y?
model/tf.math.floormod/FloorModFloorMod(model/tf.clip_by_value/clip_by_value:z:0*model/tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2!
model/tf.math.floormod/FloorMod?
model/embedding_2/CastCast(model/tf.clip_by_value/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding_2/Cast?
"model/embedding_2/embedding_lookupResourceGather'model_embedding_2_embedding_lookup_2759model/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@model/embedding_2/embedding_lookup/2759*,
_output_shapes
:??????????*
dtype02$
"model/embedding_2/embedding_lookup?
+model/embedding_2/embedding_lookup/IdentityIdentity+model/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@model/embedding_2/embedding_lookup/2759*,
_output_shapes
:??????????2-
+model/embedding_2/embedding_lookup/Identity?
-model/embedding_2/embedding_lookup/Identity_1Identity4model/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2/
-model/embedding_2/embedding_lookup/Identity_1?
model/embedding/CastCast)model/tf.compat.v1.floor_div/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding/Cast?
 model/embedding/embedding_lookupResourceGather%model_embedding_embedding_lookup_2765model/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@model/embedding/embedding_lookup/2765*,
_output_shapes
:??????????*
dtype02"
 model/embedding/embedding_lookup?
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@model/embedding/embedding_lookup/2765*,
_output_shapes
:??????????2+
)model/embedding/embedding_lookup/Identity?
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2-
+model/embedding/embedding_lookup/Identity_1?
model/tf.cast/CastCast,model/tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model/tf.cast/Cast?
 model/tf.__operators__.add/AddV2AddV26model/embedding_2/embedding_lookup/Identity_1:output:04model/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2"
 model/tf.__operators__.add/AddV2?
model/embedding_1/CastCast#model/tf.math.floormod/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding_1/Cast?
"model/embedding_1/embedding_lookupResourceGather'model_embedding_1_embedding_lookup_2773model/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@model/embedding_1/embedding_lookup/2773*,
_output_shapes
:??????????*
dtype02$
"model/embedding_1/embedding_lookup?
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@model/embedding_1/embedding_lookup/2773*,
_output_shapes
:??????????2-
+model/embedding_1/embedding_lookup/Identity?
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2/
-model/embedding_1/embedding_lookup/Identity_1?
"model/tf.__operators__.add_1/AddV2AddV2$model/tf.__operators__.add/AddV2:z:06model/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2$
"model/tf.__operators__.add_1/AddV2?
#model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model/tf.expand_dims/ExpandDims/dim?
model/tf.expand_dims/ExpandDims
ExpandDimsmodel/tf.cast/Cast:y:0,model/tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2!
model/tf.expand_dims/ExpandDims?
model/tf.math.multiply/MulMul&model/tf.__operators__.add_1/AddV2:z:0(model/tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
model/tf.math.multiply/Mul?
.model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.model/tf.math.reduce_sum/Sum/reduction_indices?
model/tf.math.reduce_sum/SumSummodel/tf.math.multiply/Mul:z:07model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
model/tf.math.reduce_sum/Sum?
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_1/Const?
model_1/flatten_1/ReshapeReshape
inputs_0_1 model_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_1/Reshape?
2model_1/tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_1/tf.clip_by_value_1/clip_by_value/Minimum/y?
0model_1/tf.clip_by_value_1/clip_by_value/MinimumMinimum"model_1/flatten_1/Reshape:output:0;model_1/tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_1/tf.clip_by_value_1/clip_by_value/Minimum?
*model_1/tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_1/tf.clip_by_value_1/clip_by_value/y?
(model_1/tf.clip_by_value_1/clip_by_valueMaximum4model_1/tf.clip_by_value_1/clip_by_value/Minimum:z:03model_1/tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_1/tf.clip_by_value_1/clip_by_value?
+model_1/tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_1/tf.compat.v1.floor_div_1/FloorDiv/y?
)model_1/tf.compat.v1.floor_div_1/FloorDivFloorDiv,model_1/tf.clip_by_value_1/clip_by_value:z:04model_1/tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_1/tf.compat.v1.floor_div_1/FloorDiv?
,model_1/tf.math.greater_equal_1/GreaterEqualGreaterEqual"model_1/flatten_1/Reshape:output:0.model_1_tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_1/tf.math.greater_equal_1/GreaterEqual?
%model_1/tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_1/tf.math.floormod_1/FloorMod/y?
#model_1/tf.math.floormod_1/FloorModFloorMod,model_1/tf.clip_by_value_1/clip_by_value:z:0.model_1/tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_1/tf.math.floormod_1/FloorMod?
model_1/embedding_5/CastCast,model_1/tf.clip_by_value_1/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_5/Cast?
$model_1/embedding_5/embedding_lookupResourceGather)model_1_embedding_5_embedding_lookup_2797model_1/embedding_5/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_1/embedding_5/embedding_lookup/2797*,
_output_shapes
:??????????*
dtype02&
$model_1/embedding_5/embedding_lookup?
-model_1/embedding_5/embedding_lookup/IdentityIdentity-model_1/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_1/embedding_5/embedding_lookup/2797*,
_output_shapes
:??????????2/
-model_1/embedding_5/embedding_lookup/Identity?
/model_1/embedding_5/embedding_lookup/Identity_1Identity6model_1/embedding_5/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????21
/model_1/embedding_5/embedding_lookup/Identity_1?
model_1/embedding_3/CastCast-model_1/tf.compat.v1.floor_div_1/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_3/Cast?
$model_1/embedding_3/embedding_lookupResourceGather)model_1_embedding_3_embedding_lookup_2803model_1/embedding_3/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_1/embedding_3/embedding_lookup/2803*,
_output_shapes
:??????????*
dtype02&
$model_1/embedding_3/embedding_lookup?
-model_1/embedding_3/embedding_lookup/IdentityIdentity-model_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_1/embedding_3/embedding_lookup/2803*,
_output_shapes
:??????????2/
-model_1/embedding_3/embedding_lookup/Identity?
/model_1/embedding_3/embedding_lookup/Identity_1Identity6model_1/embedding_3/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????21
/model_1/embedding_3/embedding_lookup/Identity_1?
model_1/tf.cast_1/CastCast0model_1/tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_1/tf.cast_1/Cast?
$model_1/tf.__operators__.add_2/AddV2AddV28model_1/embedding_5/embedding_lookup/Identity_1:output:08model_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2&
$model_1/tf.__operators__.add_2/AddV2?
model_1/embedding_4/CastCast'model_1/tf.math.floormod_1/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_4/Cast?
$model_1/embedding_4/embedding_lookupResourceGather)model_1_embedding_4_embedding_lookup_2811model_1/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*<
_class2
0.loc:@model_1/embedding_4/embedding_lookup/2811*,
_output_shapes
:??????????*
dtype02&
$model_1/embedding_4/embedding_lookup?
-model_1/embedding_4/embedding_lookup/IdentityIdentity-model_1/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@model_1/embedding_4/embedding_lookup/2811*,
_output_shapes
:??????????2/
-model_1/embedding_4/embedding_lookup/Identity?
/model_1/embedding_4/embedding_lookup/Identity_1Identity6model_1/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????21
/model_1/embedding_4/embedding_lookup/Identity_1?
$model_1/tf.__operators__.add_3/AddV2AddV2(model_1/tf.__operators__.add_2/AddV2:z:08model_1/embedding_4/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2&
$model_1/tf.__operators__.add_3/AddV2?
'model_1/tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_1/tf.expand_dims_1/ExpandDims/dim?
#model_1/tf.expand_dims_1/ExpandDims
ExpandDimsmodel_1/tf.cast_1/Cast:y:00model_1/tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_1/tf.expand_dims_1/ExpandDims?
model_1/tf.math.multiply_1/MulMul(model_1/tf.__operators__.add_3/AddV2:z:0,model_1/tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_1/tf.math.multiply_1/Mul?
2model_1/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_1/tf.math.reduce_sum_1/Sum/reduction_indices?
 model_1/tf.math.reduce_sum_1/SumSum"model_1/tf.math.multiply_1/Mul:z:0;model_1/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_1/tf.math.reduce_sum_1/Sum?
(tf.clip_by_value_2/clip_by_value/MinimumMinimuminputs_1*tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2*
(tf.clip_by_value_2/clip_by_value/Minimum?
 tf.clip_by_value_2/clip_by_valueMaximum,tf.clip_by_value_2/clip_by_value/Minimum:z:0"tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2"
 tf.clip_by_value_2/clip_by_value?
tf.cast_2/CastCast(tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_2/Castp
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat/concat/axis?
tf.concat/concatConcatV2%model/tf.math.reduce_sum/Sum:output:0)model_1/tf.math.reduce_sum_1/Sum:output:0tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat/concat}
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_1/concat/axis?
tf.concat_1/concatConcatV2$tf.clip_by_value_2/clip_by_value:z:0tf.cast_2/Cast:y:0 tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_1/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMultf.concat/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMultf.concat_1/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/BiasAdd:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAdd}
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_2/concat/axis?
tf.concat_2/concatConcatV2dense_2/Relu:activations:0dense_4/BiasAdd:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_2/concat?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMultf.concat_2/concat:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddw
tf.nn.relu/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu/Relu?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMultf.nn.relu/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAdd?
tf.__operators__.add_4/AddV2AddV2dense_6/BiasAdd:output:0tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_4/AddV2?
tf.nn.relu_1/ReluRelu tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_1/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMultf.nn.relu_1/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAdd?
tf.__operators__.add_5/AddV2AddV2dense_7/BiasAdd:output:0tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_5/AddV2?
tf.nn.relu_2/ReluRelu tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_2/Relu?
.normalize/normalization/Reshape/ReadVariableOpReadVariableOp7normalize_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype020
.normalize/normalization/Reshape/ReadVariableOp?
%normalize/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%normalize/normalization/Reshape/shape?
normalize/normalization/ReshapeReshape6normalize/normalization/Reshape/ReadVariableOp:value:0.normalize/normalization/Reshape/shape:output:0*
T0*
_output_shapes
:	?2!
normalize/normalization/Reshape?
0normalize/normalization/Reshape_1/ReadVariableOpReadVariableOp9normalize_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype022
0normalize/normalization/Reshape_1/ReadVariableOp?
'normalize/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2)
'normalize/normalization/Reshape_1/shape?
!normalize/normalization/Reshape_1Reshape8normalize/normalization/Reshape_1/ReadVariableOp:value:00normalize/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2#
!normalize/normalization/Reshape_1?
normalize/normalization/subSubtf.nn.relu_2/Relu:activations:0(normalize/normalization/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalize/normalization/sub?
normalize/normalization/SqrtSqrt*normalize/normalization/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalize/normalization/Sqrt?
!normalize/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32#
!normalize/normalization/Maximum/y?
normalize/normalization/MaximumMaximum normalize/normalization/Sqrt:y:0*normalize/normalization/Maximum/y:output:0*
T0*
_output_shapes
:	?2!
normalize/normalization/Maximum?
normalize/normalization/truedivRealDivnormalize/normalization/sub:z:0#normalize/normalization/Maximum:z:0*
T0*(
_output_shapes
:??????????2!
normalize/normalization/truediv?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMul#normalize/normalization/truediv:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdd?
IdentityIdentitydense_8/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp!^model/embedding/embedding_lookup#^model/embedding_1/embedding_lookup#^model/embedding_2/embedding_lookup%^model_1/embedding_3/embedding_lookup%^model_1/embedding_4/embedding_lookup%^model_1/embedding_5/embedding_lookup/^normalize/normalization/Reshape/ReadVariableOp1^normalize/normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2D
 model/embedding/embedding_lookup model/embedding/embedding_lookup2H
"model/embedding_1/embedding_lookup"model/embedding_1/embedding_lookup2H
"model/embedding_2/embedding_lookup"model/embedding_2/embedding_lookup2L
$model_1/embedding_3/embedding_lookup$model_1/embedding_3/embedding_lookup2L
$model_1/embedding_4/embedding_lookup$model_1/embedding_4/embedding_lookup2L
$model_1/embedding_5/embedding_lookup$model_1/embedding_5/embedding_lookup2`
.normalize/normalization/Reshape/ReadVariableOp.normalize/normalization/Reshape/ReadVariableOp2d
0normalize/normalization/Reshape_1/ReadVariableOp0normalize/normalization/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_dense_6_layer_call_and_return_conditional_losses_2147

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
?__inference_model_layer_call_and_return_conditional_losses_1606

inputs(
$tf_math_greater_equal_greaterequal_y
embedding_2_1588
embedding_1591
embedding_1_1596
identity??!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14462
flatten/PartitionedCall?
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2*
(tf.clip_by_value/clip_by_value/Minimum/y?
&tf.clip_by_value/clip_by_value/MinimumMinimum flatten/PartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/y?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!tf.compat.v1.floor_div/FloorDiv/y?
tf.compat.v1.floor_div/FloorDivFloorDiv"tf.clip_by_value/clip_by_value:z:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2!
tf.compat.v1.floor_div/FloorDiv?
"tf.math.greater_equal/GreaterEqualGreaterEqual flatten/PartitionedCall:output:0$tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2$
"tf.math.greater_equal/GreaterEqual
tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod/FloorMod/y?
tf.math.floormod/FloorModFloorMod"tf.clip_by_value/clip_by_value:z:0$tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod/FloorMod?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"tf.clip_by_value/clip_by_value:z:0embedding_2_1588*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14742%
#embedding_2/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCall#tf.compat.v1.floor_div/FloorDiv:z:0embedding_1591*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_14962#
!embedding/StatefulPartitionedCall?
tf.cast/CastCast&tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast/Cast?
tf.__operators__.add/AddV2AddV2,embedding_2/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add/AddV2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod/FloorMod:z:0embedding_1_1596*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15202%
#embedding_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0,embedding_1/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_1/AddV2?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimstf.cast/Cast:y:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims/ExpandDims?
tf.math.multiply/MulMul tf.__operators__.add_1/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply/Mul?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum/Sum?
IdentityIdentitytf.math.reduce_sum/Sum:output:0"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
A__inference_dense_7_layer_call_and_return_conditional_losses_2175

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_3_layer_call_and_return_conditional_losses_1722

inputs
embedding_lookup_1716
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_1716Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/1716*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1716*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_custom_model_layer_call_fn_3217

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_custom_model_layer_call_and_return_conditional_losses_26032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_dense_1_layer_call_fn_3477

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_20392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_2_layer_call_and_return_conditional_losses_3658

inputs
embedding_lookup_3652
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3652Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/3652*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3652*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_5_layer_call_and_return_conditional_losses_3545

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_1_layer_call_and_return_conditional_losses_3692

inputs
embedding_lookup_3686
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3686Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/3686*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3686*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_8_layer_call_and_return_conditional_losses_2236

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_3448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_3_layer_call_and_return_conditional_losses_3487

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_1617
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?	
?
A__inference_dense_4_layer_call_and_return_conditional_losses_3526

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_3_layer_call_and_return_conditional_losses_2012

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
?__inference_model_layer_call_and_return_conditional_losses_1571
input_1(
$tf_math_greater_equal_greaterequal_y
embedding_2_1553
embedding_1556
embedding_1_1561
identity??!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14462
flatten/PartitionedCall?
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2*
(tf.clip_by_value/clip_by_value/Minimum/y?
&tf.clip_by_value/clip_by_value/MinimumMinimum flatten/PartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/y?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!tf.compat.v1.floor_div/FloorDiv/y?
tf.compat.v1.floor_div/FloorDivFloorDiv"tf.clip_by_value/clip_by_value:z:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2!
tf.compat.v1.floor_div/FloorDiv?
"tf.math.greater_equal/GreaterEqualGreaterEqual flatten/PartitionedCall:output:0$tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2$
"tf.math.greater_equal/GreaterEqual
tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod/FloorMod/y?
tf.math.floormod/FloorModFloorMod"tf.clip_by_value/clip_by_value:z:0$tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod/FloorMod?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"tf.clip_by_value/clip_by_value:z:0embedding_2_1553*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14742%
#embedding_2/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCall#tf.compat.v1.floor_div/FloorDiv:z:0embedding_1556*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_14962#
!embedding/StatefulPartitionedCall?
tf.cast/CastCast&tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast/Cast?
tf.__operators__.add/AddV2AddV2,embedding_2/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add/AddV2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod/FloorMod:z:0embedding_1_1561*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15202%
#embedding_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0,embedding_1/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_1/AddV2?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimstf.cast/Cast:y:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims/ExpandDims?
tf.math.multiply/MulMul tf.__operators__.add_1/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply/Mul?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum/Sum?
IdentityIdentitytf.math.reduce_sum/Sum:output:0"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
{
&__inference_dense_4_layer_call_fn_3535

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_20922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
A__inference_model_1_layer_call_and_return_conditional_losses_3411

inputs*
&tf_math_greater_equal_1_greaterequal_y%
!embedding_5_embedding_lookup_3385%
!embedding_3_embedding_lookup_3391%
!embedding_4_embedding_lookup_3399
identity??embedding_3/embedding_lookup?embedding_4/embedding_lookup?embedding_5/embedding_lookups
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeinputsflatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_1/Reshape?
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_1/clip_by_value/Minimum/y?
(tf.clip_by_value_1/clip_by_value/MinimumMinimumflatten_1/Reshape:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_1/clip_by_value/Minimum?
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_1/clip_by_value/y?
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_1/clip_by_value?
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_1/FloorDiv/y?
!tf.compat.v1.floor_div_1/FloorDivFloorDiv$tf.clip_by_value_1/clip_by_value:z:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_1/FloorDiv?
$tf.math.greater_equal_1/GreaterEqualGreaterEqualflatten_1/Reshape:output:0&tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_1/GreaterEqual?
tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_1/FloorMod/y?
tf.math.floormod_1/FloorModFloorMod$tf.clip_by_value_1/clip_by_value:z:0&tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_1/FloorMod?
embedding_5/CastCast$tf.clip_by_value_1/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_5/Cast?
embedding_5/embedding_lookupResourceGather!embedding_5_embedding_lookup_3385embedding_5/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_5/embedding_lookup/3385*,
_output_shapes
:??????????*
dtype02
embedding_5/embedding_lookup?
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_5/embedding_lookup/3385*,
_output_shapes
:??????????2'
%embedding_5/embedding_lookup/Identity?
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_5/embedding_lookup/Identity_1?
embedding_3/CastCast%tf.compat.v1.floor_div_1/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_3/Cast?
embedding_3/embedding_lookupResourceGather!embedding_3_embedding_lookup_3391embedding_3/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_3/embedding_lookup/3391*,
_output_shapes
:??????????*
dtype02
embedding_3/embedding_lookup?
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_3/embedding_lookup/3391*,
_output_shapes
:??????????2'
%embedding_3/embedding_lookup/Identity?
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_3/embedding_lookup/Identity_1?
tf.cast_1/CastCast(tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_1/Cast?
tf.__operators__.add_2/AddV2AddV20embedding_5/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_2/AddV2?
embedding_4/CastCasttf.math.floormod_1/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_4/Cast?
embedding_4/embedding_lookupResourceGather!embedding_4_embedding_lookup_3399embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_4/embedding_lookup/3399*,
_output_shapes
:??????????*
dtype02
embedding_4/embedding_lookup?
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_4/embedding_lookup/3399*,
_output_shapes
:??????????2'
%embedding_4/embedding_lookup/Identity?
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_4/embedding_lookup/Identity_1?
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:00embedding_4/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_3/AddV2?
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_1/ExpandDims/dim?
tf.expand_dims_1/ExpandDims
ExpandDimstf.cast_1/Cast:y:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_1/ExpandDims?
tf.math.multiply_1/MulMul tf.__operators__.add_3/AddV2:z:0$tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_1/Mul?
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_1/Sum/reduction_indices?
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_1/Sum?
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^embedding_3/embedding_lookup^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_3507

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?,
?
?__inference_model_layer_call_and_return_conditional_losses_1539
input_1(
$tf_math_greater_equal_greaterequal_y
embedding_2_1483
embedding_1505
embedding_1_1529
identity??!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14462
flatten/PartitionedCall?
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2*
(tf.clip_by_value/clip_by_value/Minimum/y?
&tf.clip_by_value/clip_by_value/MinimumMinimum flatten/PartitionedCall:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/y?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!tf.compat.v1.floor_div/FloorDiv/y?
tf.compat.v1.floor_div/FloorDivFloorDiv"tf.clip_by_value/clip_by_value:z:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2!
tf.compat.v1.floor_div/FloorDiv?
"tf.math.greater_equal/GreaterEqualGreaterEqual flatten/PartitionedCall:output:0$tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2$
"tf.math.greater_equal/GreaterEqual
tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod/FloorMod/y?
tf.math.floormod/FloorModFloorMod"tf.clip_by_value/clip_by_value:z:0$tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod/FloorMod?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall"tf.clip_by_value/clip_by_value:z:0embedding_2_1483*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14742%
#embedding_2/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCall#tf.compat.v1.floor_div/FloorDiv:z:0embedding_1505*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_14962#
!embedding/StatefulPartitionedCall?
tf.cast/CastCast&tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast/Cast?
tf.__operators__.add/AddV2AddV2,embedding_2/StatefulPartitionedCall:output:0*embedding/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add/AddV2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod/FloorMod:z:0embedding_1_1529*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15202%
#embedding_1/StatefulPartitionedCall?
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:0,embedding_1/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_1/AddV2?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimstf.cast/Cast:y:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims/ExpandDims?
tf.math.multiply/MulMul tf.__operators__.add_1/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply/Mul?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum/Sum?
IdentityIdentitytf.math.reduce_sum/Sum:output:0"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: 
?
?
+__inference_custom_model_layer_call_fn_2668

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_custom_model_layer_call_and_return_conditional_losses_26032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?6
?
?__inference_model_layer_call_and_return_conditional_losses_3301

inputs(
$tf_math_greater_equal_greaterequal_y%
!embedding_2_embedding_lookup_3275#
embedding_embedding_lookup_3281%
!embedding_1_embedding_lookup_3289
identity??embedding/embedding_lookup?embedding_1/embedding_lookup?embedding_2/embedding_lookupo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten/Reshape?
(tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2*
(tf.clip_by_value/clip_by_value/Minimum/y?
&tf.clip_by_value/clip_by_value/MinimumMinimumflatten/Reshape:output:01tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
 tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 tf.clip_by_value/clip_by_value/y?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0)tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
!tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!tf.compat.v1.floor_div/FloorDiv/y?
tf.compat.v1.floor_div/FloorDivFloorDiv"tf.clip_by_value/clip_by_value:z:0*tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2!
tf.compat.v1.floor_div/FloorDiv?
"tf.math.greater_equal/GreaterEqualGreaterEqualflatten/Reshape:output:0$tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????2$
"tf.math.greater_equal/GreaterEqual
tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod/FloorMod/y?
tf.math.floormod/FloorModFloorMod"tf.clip_by_value/clip_by_value:z:0$tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod/FloorMod?
embedding_2/CastCast"tf.clip_by_value/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_2/Cast?
embedding_2/embedding_lookupResourceGather!embedding_2_embedding_lookup_3275embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_2/embedding_lookup/3275*,
_output_shapes
:??????????*
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_2/embedding_lookup/3275*,
_output_shapes
:??????????2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_2/embedding_lookup/Identity_1?
embedding/CastCast#tf.compat.v1.floor_div/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_3281embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/3281*,
_output_shapes
:??????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/3281*,
_output_shapes
:??????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2'
%embedding/embedding_lookup/Identity_1?
tf.cast/CastCast&tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast/Cast?
tf.__operators__.add/AddV2AddV20embedding_2/embedding_lookup/Identity_1:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add/AddV2?
embedding_1/CastCasttf.math.floormod/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_3289embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_1/embedding_lookup/3289*,
_output_shapes
:??????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/3289*,
_output_shapes
:??????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_1/embedding_lookup/Identity_1?
tf.__operators__.add_1/AddV2AddV2tf.__operators__.add/AddV2:z:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_1/AddV2?
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.expand_dims/ExpandDims/dim?
tf.expand_dims/ExpandDims
ExpandDimstf.cast/Cast:y:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims/ExpandDims?
tf.math.multiply/MulMul tf.__operators__.add_1/AddV2:z:0"tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply/Mul?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum/Sum?
IdentityIdentitytf.math.reduce_sum/Sum:output:0^embedding/embedding_lookup^embedding_1/embedding_lookup^embedding_2/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2<
embedding_2/embedding_lookupembedding_2/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
{
&__inference_dense_2_layer_call_fn_3516

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_20662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
x
(__inference_normalize_layer_call_fn_3618
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_normalize_layer_call_and_return_conditional_losses_22102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_3705

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_flatten_layer_call_fn_3648

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_2_layer_call_and_return_conditional_losses_1474

inputs
embedding_lookup_1468
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_1468Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/1468*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1468*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_1672

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_2739
bets

cards0

cards1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *(
f#R!
__inference__wrapped_model_14362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????
:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????


_user_specified_namebets:OK
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_custom_model_layer_call_fn_2507

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_custom_model_layer_call_and_return_conditional_losses_24422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
E__inference_embedding_5_layer_call_and_return_conditional_losses_1700

inputs
embedding_lookup_1694
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_1694Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/1694*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1694*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
A__inference_model_1_layer_call_and_return_conditional_losses_1832

inputs*
&tf_math_greater_equal_1_greaterequal_y
embedding_5_1814
embedding_3_1817
embedding_4_1822
identity??#embedding_3/StatefulPartitionedCall?#embedding_4/StatefulPartitionedCall?#embedding_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_16722
flatten_1/PartitionedCall?
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_1/clip_by_value/Minimum/y?
(tf.clip_by_value_1/clip_by_value/MinimumMinimum"flatten_1/PartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_1/clip_by_value/Minimum?
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_1/clip_by_value/y?
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_1/clip_by_value?
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_1/FloorDiv/y?
!tf.compat.v1.floor_div_1/FloorDivFloorDiv$tf.clip_by_value_1/clip_by_value:z:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_1/FloorDiv?
$tf.math.greater_equal_1/GreaterEqualGreaterEqual"flatten_1/PartitionedCall:output:0&tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_1/GreaterEqual?
tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_1/FloorMod/y?
tf.math.floormod_1/FloorModFloorMod$tf.clip_by_value_1/clip_by_value:z:0&tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_1/FloorMod?
#embedding_5/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_1/clip_by_value:z:0embedding_5_1814*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_5_layer_call_and_return_conditional_losses_17002%
#embedding_5/StatefulPartitionedCall?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_1/FloorDiv:z:0embedding_3_1817*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_17222%
#embedding_3/StatefulPartitionedCall?
tf.cast_1/CastCast(tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_1/Cast?
tf.__operators__.add_2/AddV2AddV2,embedding_5/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_2/AddV2?
#embedding_4/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_1/FloorMod:z:0embedding_4_1822*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_17462%
#embedding_4/StatefulPartitionedCall?
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0,embedding_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_3/AddV2?
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_1/ExpandDims/dim?
tf.expand_dims_1/ExpandDims
ExpandDimstf.cast_1/Cast:y:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_1/ExpandDims?
tf.math.multiply_1/MulMul tf.__operators__.add_3/AddV2:z:0$tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_1/Mul?
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_1/Sum/reduction_indices?
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_1/Sum?
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
D
(__inference_flatten_1_layer_call_fn_3710

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_16722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense_3_layer_call_fn_3496

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_20122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
*__inference_embedding_4_layer_call_fn_3761

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_17462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense_5_layer_call_fn_3554

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_21202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_2066

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_3_layer_call_and_return_conditional_losses_3737

inputs
embedding_lookup_3731
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3731Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/3731*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3731*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_dense_7_layer_call_fn_3592

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_21752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_3424

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?8
?
A__inference_model_1_layer_call_and_return_conditional_losses_3369

inputs*
&tf_math_greater_equal_1_greaterequal_y%
!embedding_5_embedding_lookup_3343%
!embedding_3_embedding_lookup_3349%
!embedding_4_embedding_lookup_3357
identity??embedding_3/embedding_lookup?embedding_4/embedding_lookup?embedding_5/embedding_lookups
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_1/Const?
flatten_1/ReshapeReshapeinputsflatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_1/Reshape?
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_1/clip_by_value/Minimum/y?
(tf.clip_by_value_1/clip_by_value/MinimumMinimumflatten_1/Reshape:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_1/clip_by_value/Minimum?
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_1/clip_by_value/y?
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_1/clip_by_value?
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_1/FloorDiv/y?
!tf.compat.v1.floor_div_1/FloorDivFloorDiv$tf.clip_by_value_1/clip_by_value:z:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_1/FloorDiv?
$tf.math.greater_equal_1/GreaterEqualGreaterEqualflatten_1/Reshape:output:0&tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_1/GreaterEqual?
tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_1/FloorMod/y?
tf.math.floormod_1/FloorModFloorMod$tf.clip_by_value_1/clip_by_value:z:0&tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_1/FloorMod?
embedding_5/CastCast$tf.clip_by_value_1/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_5/Cast?
embedding_5/embedding_lookupResourceGather!embedding_5_embedding_lookup_3343embedding_5/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_5/embedding_lookup/3343*,
_output_shapes
:??????????*
dtype02
embedding_5/embedding_lookup?
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_5/embedding_lookup/3343*,
_output_shapes
:??????????2'
%embedding_5/embedding_lookup/Identity?
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_5/embedding_lookup/Identity_1?
embedding_3/CastCast%tf.compat.v1.floor_div_1/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_3/Cast?
embedding_3/embedding_lookupResourceGather!embedding_3_embedding_lookup_3349embedding_3/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_3/embedding_lookup/3349*,
_output_shapes
:??????????*
dtype02
embedding_3/embedding_lookup?
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_3/embedding_lookup/3349*,
_output_shapes
:??????????2'
%embedding_3/embedding_lookup/Identity?
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_3/embedding_lookup/Identity_1?
tf.cast_1/CastCast(tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_1/Cast?
tf.__operators__.add_2/AddV2AddV20embedding_5/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_2/AddV2?
embedding_4/CastCasttf.math.floormod_1/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_4/Cast?
embedding_4/embedding_lookupResourceGather!embedding_4_embedding_lookup_3357embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding_4/embedding_lookup/3357*,
_output_shapes
:??????????*
dtype02
embedding_4/embedding_lookup?
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_4/embedding_lookup/3357*,
_output_shapes
:??????????2'
%embedding_4/embedding_lookup/Identity?
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2)
'embedding_4/embedding_lookup/Identity_1?
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:00embedding_4/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_3/AddV2?
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_1/ExpandDims/dim?
tf.expand_dims_1/ExpandDims
ExpandDimstf.cast_1/Cast:y:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_1/ExpandDims?
tf.math.multiply_1/MulMul tf.__operators__.add_3/AddV2:z:0$tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_1/Mul?
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_1/Sum/reduction_indices?
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_1/Sum?
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0^embedding_3/embedding_lookup^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
A__inference_dense_4_layer_call_and_return_conditional_losses_2092

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_7_layer_call_and_return_conditional_losses_3583

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_custom_model_layer_call_fn_3148

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_custom_model_layer_call_and_return_conditional_losses_24422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_dense_1_layer_call_and_return_conditional_losses_3468

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_embedding_4_layer_call_and_return_conditional_losses_3754

inputs
embedding_lookup_3748
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3748Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/3748*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/3748*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
A__inference_model_1_layer_call_and_return_conditional_losses_1877

inputs*
&tf_math_greater_equal_1_greaterequal_y
embedding_5_1859
embedding_3_1862
embedding_4_1867
identity??#embedding_3/StatefulPartitionedCall?#embedding_4/StatefulPartitionedCall?#embedding_5/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_16722
flatten_1/PartitionedCall?
*tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_1/clip_by_value/Minimum/y?
(tf.clip_by_value_1/clip_by_value/MinimumMinimum"flatten_1/PartitionedCall:output:03tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_1/clip_by_value/Minimum?
"tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_1/clip_by_value/y?
 tf.clip_by_value_1/clip_by_valueMaximum,tf.clip_by_value_1/clip_by_value/Minimum:z:0+tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_1/clip_by_value?
#tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_1/FloorDiv/y?
!tf.compat.v1.floor_div_1/FloorDivFloorDiv$tf.clip_by_value_1/clip_by_value:z:0,tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_1/FloorDiv?
$tf.math.greater_equal_1/GreaterEqualGreaterEqual"flatten_1/PartitionedCall:output:0&tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_1/GreaterEqual?
tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_1/FloorMod/y?
tf.math.floormod_1/FloorModFloorMod$tf.clip_by_value_1/clip_by_value:z:0&tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_1/FloorMod?
#embedding_5/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_1/clip_by_value:z:0embedding_5_1859*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_5_layer_call_and_return_conditional_losses_17002%
#embedding_5/StatefulPartitionedCall?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_1/FloorDiv:z:0embedding_3_1862*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_17222%
#embedding_3/StatefulPartitionedCall?
tf.cast_1/CastCast(tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_1/Cast?
tf.__operators__.add_2/AddV2AddV2,embedding_5/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_2/AddV2?
#embedding_4/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_1/FloorMod:z:0embedding_4_1867*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_17462%
#embedding_4/StatefulPartitionedCall?
tf.__operators__.add_3/AddV2AddV2 tf.__operators__.add_2/AddV2:z:0,embedding_4/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_3/AddV2?
tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_1/ExpandDims/dim?
tf.expand_dims_1/ExpandDims
ExpandDimstf.cast_1/Cast:y:0(tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_1/ExpandDims?
tf.math.multiply_1/MulMul tf.__operators__.add_3/AddV2:z:0$tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_1/Mul?
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_1/Sum/reduction_indices?
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_1/Sum?
IdentityIdentity!tf.math.reduce_sum_1/Sum:output:0$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
C__inference_normalize_layer_call_and_return_conditional_losses_2210
x1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization/Reshape_1?
normalization/subSubxnormalization/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization/sub|
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization/truediv?
IdentityIdentitynormalization/truediv:z:0%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
Ԩ
?
__inference__wrapped_model_1436

cards0

cards1
bets7
3custom_model_tf_math_greater_equal_2_greaterequal_y;
7custom_model_model_tf_math_greater_equal_greaterequal_y8
4custom_model_model_embedding_2_embedding_lookup_12866
2custom_model_model_embedding_embedding_lookup_12928
4custom_model_model_embedding_1_embedding_lookup_1300?
;custom_model_model_1_tf_math_greater_equal_1_greaterequal_y:
6custom_model_model_1_embedding_5_embedding_lookup_1324:
6custom_model_model_1_embedding_3_embedding_lookup_1330:
6custom_model_model_1_embedding_4_embedding_lookup_1338;
7custom_model_tf_clip_by_value_2_clip_by_value_minimum_y3
/custom_model_tf_clip_by_value_2_clip_by_value_y5
1custom_model_dense_matmul_readvariableop_resource6
2custom_model_dense_biasadd_readvariableop_resource7
3custom_model_dense_3_matmul_readvariableop_resource8
4custom_model_dense_3_biasadd_readvariableop_resource7
3custom_model_dense_1_matmul_readvariableop_resource8
4custom_model_dense_1_biasadd_readvariableop_resource7
3custom_model_dense_2_matmul_readvariableop_resource8
4custom_model_dense_2_biasadd_readvariableop_resource7
3custom_model_dense_4_matmul_readvariableop_resource8
4custom_model_dense_4_biasadd_readvariableop_resource7
3custom_model_dense_5_matmul_readvariableop_resource8
4custom_model_dense_5_biasadd_readvariableop_resource7
3custom_model_dense_6_matmul_readvariableop_resource8
4custom_model_dense_6_biasadd_readvariableop_resource7
3custom_model_dense_7_matmul_readvariableop_resource8
4custom_model_dense_7_biasadd_readvariableop_resourceH
Dcustom_model_normalize_normalization_reshape_readvariableop_resourceJ
Fcustom_model_normalize_normalization_reshape_1_readvariableop_resource7
3custom_model_dense_8_matmul_readvariableop_resource8
4custom_model_dense_8_biasadd_readvariableop_resource
identity??)custom_model/dense/BiasAdd/ReadVariableOp?(custom_model/dense/MatMul/ReadVariableOp?+custom_model/dense_1/BiasAdd/ReadVariableOp?*custom_model/dense_1/MatMul/ReadVariableOp?+custom_model/dense_2/BiasAdd/ReadVariableOp?*custom_model/dense_2/MatMul/ReadVariableOp?+custom_model/dense_3/BiasAdd/ReadVariableOp?*custom_model/dense_3/MatMul/ReadVariableOp?+custom_model/dense_4/BiasAdd/ReadVariableOp?*custom_model/dense_4/MatMul/ReadVariableOp?+custom_model/dense_5/BiasAdd/ReadVariableOp?*custom_model/dense_5/MatMul/ReadVariableOp?+custom_model/dense_6/BiasAdd/ReadVariableOp?*custom_model/dense_6/MatMul/ReadVariableOp?+custom_model/dense_7/BiasAdd/ReadVariableOp?*custom_model/dense_7/MatMul/ReadVariableOp?+custom_model/dense_8/BiasAdd/ReadVariableOp?*custom_model/dense_8/MatMul/ReadVariableOp?-custom_model/model/embedding/embedding_lookup?/custom_model/model/embedding_1/embedding_lookup?/custom_model/model/embedding_2/embedding_lookup?1custom_model/model_1/embedding_3/embedding_lookup?1custom_model/model_1/embedding_4/embedding_lookup?1custom_model/model_1/embedding_5/embedding_lookup?;custom_model/normalize/normalization/Reshape/ReadVariableOp?=custom_model/normalize/normalization/Reshape_1/ReadVariableOp?
1custom_model/tf.math.greater_equal_2/GreaterEqualGreaterEqualbets3custom_model_tf_math_greater_equal_2_greaterequal_y*
T0*'
_output_shapes
:?????????
23
1custom_model/tf.math.greater_equal_2/GreaterEqual?
 custom_model/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2"
 custom_model/model/flatten/Const?
"custom_model/model/flatten/ReshapeReshapecards0)custom_model/model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2$
"custom_model/model/flatten/Reshape?
;custom_model/model/tf.clip_by_value/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2=
;custom_model/model/tf.clip_by_value/clip_by_value/Minimum/y?
9custom_model/model/tf.clip_by_value/clip_by_value/MinimumMinimum+custom_model/model/flatten/Reshape:output:0Dcustom_model/model/tf.clip_by_value/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2;
9custom_model/model/tf.clip_by_value/clip_by_value/Minimum?
3custom_model/model/tf.clip_by_value/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3custom_model/model/tf.clip_by_value/clip_by_value/y?
1custom_model/model/tf.clip_by_value/clip_by_valueMaximum=custom_model/model/tf.clip_by_value/clip_by_value/Minimum:z:0<custom_model/model/tf.clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????23
1custom_model/model/tf.clip_by_value/clip_by_value?
4custom_model/model/tf.compat.v1.floor_div/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model/model/tf.compat.v1.floor_div/FloorDiv/y?
2custom_model/model/tf.compat.v1.floor_div/FloorDivFloorDiv5custom_model/model/tf.clip_by_value/clip_by_value:z:0=custom_model/model/tf.compat.v1.floor_div/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model/model/tf.compat.v1.floor_div/FloorDiv?
5custom_model/model/tf.math.greater_equal/GreaterEqualGreaterEqual+custom_model/model/flatten/Reshape:output:07custom_model_model_tf_math_greater_equal_greaterequal_y*
T0*'
_output_shapes
:?????????27
5custom_model/model/tf.math.greater_equal/GreaterEqual?
.custom_model/model/tf.math.floormod/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@20
.custom_model/model/tf.math.floormod/FloorMod/y?
,custom_model/model/tf.math.floormod/FloorModFloorMod5custom_model/model/tf.clip_by_value/clip_by_value:z:07custom_model/model/tf.math.floormod/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2.
,custom_model/model/tf.math.floormod/FloorMod?
#custom_model/model/embedding_2/CastCast5custom_model/model/tf.clip_by_value/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2%
#custom_model/model/embedding_2/Cast?
/custom_model/model/embedding_2/embedding_lookupResourceGather4custom_model_model_embedding_2_embedding_lookup_1286'custom_model/model/embedding_2/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*G
_class=
;9loc:@custom_model/model/embedding_2/embedding_lookup/1286*,
_output_shapes
:??????????*
dtype021
/custom_model/model/embedding_2/embedding_lookup?
8custom_model/model/embedding_2/embedding_lookup/IdentityIdentity8custom_model/model/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/model/embedding_2/embedding_lookup/1286*,
_output_shapes
:??????????2:
8custom_model/model/embedding_2/embedding_lookup/Identity?
:custom_model/model/embedding_2/embedding_lookup/Identity_1IdentityAcustom_model/model/embedding_2/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2<
:custom_model/model/embedding_2/embedding_lookup/Identity_1?
!custom_model/model/embedding/CastCast6custom_model/model/tf.compat.v1.floor_div/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2#
!custom_model/model/embedding/Cast?
-custom_model/model/embedding/embedding_lookupResourceGather2custom_model_model_embedding_embedding_lookup_1292%custom_model/model/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*E
_class;
97loc:@custom_model/model/embedding/embedding_lookup/1292*,
_output_shapes
:??????????*
dtype02/
-custom_model/model/embedding/embedding_lookup?
6custom_model/model/embedding/embedding_lookup/IdentityIdentity6custom_model/model/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@custom_model/model/embedding/embedding_lookup/1292*,
_output_shapes
:??????????28
6custom_model/model/embedding/embedding_lookup/Identity?
8custom_model/model/embedding/embedding_lookup/Identity_1Identity?custom_model/model/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2:
8custom_model/model/embedding/embedding_lookup/Identity_1?
custom_model/model/tf.cast/CastCast9custom_model/model/tf.math.greater_equal/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2!
custom_model/model/tf.cast/Cast?
-custom_model/model/tf.__operators__.add/AddV2AddV2Ccustom_model/model/embedding_2/embedding_lookup/Identity_1:output:0Acustom_model/model/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model/model/tf.__operators__.add/AddV2?
#custom_model/model/embedding_1/CastCast0custom_model/model/tf.math.floormod/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2%
#custom_model/model/embedding_1/Cast?
/custom_model/model/embedding_1/embedding_lookupResourceGather4custom_model_model_embedding_1_embedding_lookup_1300'custom_model/model/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*G
_class=
;9loc:@custom_model/model/embedding_1/embedding_lookup/1300*,
_output_shapes
:??????????*
dtype021
/custom_model/model/embedding_1/embedding_lookup?
8custom_model/model/embedding_1/embedding_lookup/IdentityIdentity8custom_model/model/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@custom_model/model/embedding_1/embedding_lookup/1300*,
_output_shapes
:??????????2:
8custom_model/model/embedding_1/embedding_lookup/Identity?
:custom_model/model/embedding_1/embedding_lookup/Identity_1IdentityAcustom_model/model/embedding_1/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2<
:custom_model/model/embedding_1/embedding_lookup/Identity_1?
/custom_model/model/tf.__operators__.add_1/AddV2AddV21custom_model/model/tf.__operators__.add/AddV2:z:0Ccustom_model/model/embedding_1/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????21
/custom_model/model/tf.__operators__.add_1/AddV2?
0custom_model/model/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0custom_model/model/tf.expand_dims/ExpandDims/dim?
,custom_model/model/tf.expand_dims/ExpandDims
ExpandDims#custom_model/model/tf.cast/Cast:y:09custom_model/model/tf.expand_dims/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2.
,custom_model/model/tf.expand_dims/ExpandDims?
'custom_model/model/tf.math.multiply/MulMul3custom_model/model/tf.__operators__.add_1/AddV2:z:05custom_model/model/tf.expand_dims/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2)
'custom_model/model/tf.math.multiply/Mul?
;custom_model/model/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;custom_model/model/tf.math.reduce_sum/Sum/reduction_indices?
)custom_model/model/tf.math.reduce_sum/SumSum+custom_model/model/tf.math.multiply/Mul:z:0Dcustom_model/model/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2+
)custom_model/model/tf.math.reduce_sum/Sum?
$custom_model/model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$custom_model/model_1/flatten_1/Const?
&custom_model/model_1/flatten_1/ReshapeReshapecards1-custom_model/model_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????2(
&custom_model/model_1/flatten_1/Reshape?
?custom_model/model_1/tf.clip_by_value_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2A
?custom_model/model_1/tf.clip_by_value_1/clip_by_value/Minimum/y?
=custom_model/model_1/tf.clip_by_value_1/clip_by_value/MinimumMinimum/custom_model/model_1/flatten_1/Reshape:output:0Hcustom_model/model_1/tf.clip_by_value_1/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2?
=custom_model/model_1/tf.clip_by_value_1/clip_by_value/Minimum?
7custom_model/model_1/tf.clip_by_value_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7custom_model/model_1/tf.clip_by_value_1/clip_by_value/y?
5custom_model/model_1/tf.clip_by_value_1/clip_by_valueMaximumAcustom_model/model_1/tf.clip_by_value_1/clip_by_value/Minimum:z:0@custom_model/model_1/tf.clip_by_value_1/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????27
5custom_model/model_1/tf.clip_by_value_1/clip_by_value?
8custom_model/model_1/tf.compat.v1.floor_div_1/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2:
8custom_model/model_1/tf.compat.v1.floor_div_1/FloorDiv/y?
6custom_model/model_1/tf.compat.v1.floor_div_1/FloorDivFloorDiv9custom_model/model_1/tf.clip_by_value_1/clip_by_value:z:0Acustom_model/model_1/tf.compat.v1.floor_div_1/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????28
6custom_model/model_1/tf.compat.v1.floor_div_1/FloorDiv?
9custom_model/model_1/tf.math.greater_equal_1/GreaterEqualGreaterEqual/custom_model/model_1/flatten_1/Reshape:output:0;custom_model_model_1_tf_math_greater_equal_1_greaterequal_y*
T0*'
_output_shapes
:?????????2;
9custom_model/model_1/tf.math.greater_equal_1/GreaterEqual?
2custom_model/model_1/tf.math.floormod_1/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@24
2custom_model/model_1/tf.math.floormod_1/FloorMod/y?
0custom_model/model_1/tf.math.floormod_1/FloorModFloorMod9custom_model/model_1/tf.clip_by_value_1/clip_by_value:z:0;custom_model/model_1/tf.math.floormod_1/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????22
0custom_model/model_1/tf.math.floormod_1/FloorMod?
%custom_model/model_1/embedding_5/CastCast9custom_model/model_1/tf.clip_by_value_1/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2'
%custom_model/model_1/embedding_5/Cast?
1custom_model/model_1/embedding_5/embedding_lookupResourceGather6custom_model_model_1_embedding_5_embedding_lookup_1324)custom_model/model_1/embedding_5/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*I
_class?
=;loc:@custom_model/model_1/embedding_5/embedding_lookup/1324*,
_output_shapes
:??????????*
dtype023
1custom_model/model_1/embedding_5/embedding_lookup?
:custom_model/model_1/embedding_5/embedding_lookup/IdentityIdentity:custom_model/model_1/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@custom_model/model_1/embedding_5/embedding_lookup/1324*,
_output_shapes
:??????????2<
:custom_model/model_1/embedding_5/embedding_lookup/Identity?
<custom_model/model_1/embedding_5/embedding_lookup/Identity_1IdentityCcustom_model/model_1/embedding_5/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2>
<custom_model/model_1/embedding_5/embedding_lookup/Identity_1?
%custom_model/model_1/embedding_3/CastCast:custom_model/model_1/tf.compat.v1.floor_div_1/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2'
%custom_model/model_1/embedding_3/Cast?
1custom_model/model_1/embedding_3/embedding_lookupResourceGather6custom_model_model_1_embedding_3_embedding_lookup_1330)custom_model/model_1/embedding_3/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*I
_class?
=;loc:@custom_model/model_1/embedding_3/embedding_lookup/1330*,
_output_shapes
:??????????*
dtype023
1custom_model/model_1/embedding_3/embedding_lookup?
:custom_model/model_1/embedding_3/embedding_lookup/IdentityIdentity:custom_model/model_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@custom_model/model_1/embedding_3/embedding_lookup/1330*,
_output_shapes
:??????????2<
:custom_model/model_1/embedding_3/embedding_lookup/Identity?
<custom_model/model_1/embedding_3/embedding_lookup/Identity_1IdentityCcustom_model/model_1/embedding_3/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2>
<custom_model/model_1/embedding_3/embedding_lookup/Identity_1?
#custom_model/model_1/tf.cast_1/CastCast=custom_model/model_1/tf.math.greater_equal_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2%
#custom_model/model_1/tf.cast_1/Cast?
1custom_model/model_1/tf.__operators__.add_2/AddV2AddV2Ecustom_model/model_1/embedding_5/embedding_lookup/Identity_1:output:0Ecustom_model/model_1/embedding_3/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????23
1custom_model/model_1/tf.__operators__.add_2/AddV2?
%custom_model/model_1/embedding_4/CastCast4custom_model/model_1/tf.math.floormod_1/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2'
%custom_model/model_1/embedding_4/Cast?
1custom_model/model_1/embedding_4/embedding_lookupResourceGather6custom_model_model_1_embedding_4_embedding_lookup_1338)custom_model/model_1/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*I
_class?
=;loc:@custom_model/model_1/embedding_4/embedding_lookup/1338*,
_output_shapes
:??????????*
dtype023
1custom_model/model_1/embedding_4/embedding_lookup?
:custom_model/model_1/embedding_4/embedding_lookup/IdentityIdentity:custom_model/model_1/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@custom_model/model_1/embedding_4/embedding_lookup/1338*,
_output_shapes
:??????????2<
:custom_model/model_1/embedding_4/embedding_lookup/Identity?
<custom_model/model_1/embedding_4/embedding_lookup/Identity_1IdentityCcustom_model/model_1/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2>
<custom_model/model_1/embedding_4/embedding_lookup/Identity_1?
1custom_model/model_1/tf.__operators__.add_3/AddV2AddV25custom_model/model_1/tf.__operators__.add_2/AddV2:z:0Ecustom_model/model_1/embedding_4/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????23
1custom_model/model_1/tf.__operators__.add_3/AddV2?
4custom_model/model_1/tf.expand_dims_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4custom_model/model_1/tf.expand_dims_1/ExpandDims/dim?
0custom_model/model_1/tf.expand_dims_1/ExpandDims
ExpandDims'custom_model/model_1/tf.cast_1/Cast:y:0=custom_model/model_1/tf.expand_dims_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????22
0custom_model/model_1/tf.expand_dims_1/ExpandDims?
+custom_model/model_1/tf.math.multiply_1/MulMul5custom_model/model_1/tf.__operators__.add_3/AddV2:z:09custom_model/model_1/tf.expand_dims_1/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2-
+custom_model/model_1/tf.math.multiply_1/Mul?
?custom_model/model_1/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2A
?custom_model/model_1/tf.math.reduce_sum_1/Sum/reduction_indices?
-custom_model/model_1/tf.math.reduce_sum_1/SumSum/custom_model/model_1/tf.math.multiply_1/Mul:z:0Hcustom_model/model_1/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2/
-custom_model/model_1/tf.math.reduce_sum_1/Sum?
5custom_model/tf.clip_by_value_2/clip_by_value/MinimumMinimumbets7custom_model_tf_clip_by_value_2_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
27
5custom_model/tf.clip_by_value_2/clip_by_value/Minimum?
-custom_model/tf.clip_by_value_2/clip_by_valueMaximum9custom_model/tf.clip_by_value_2/clip_by_value/Minimum:z:0/custom_model_tf_clip_by_value_2_clip_by_value_y*
T0*'
_output_shapes
:?????????
2/
-custom_model/tf.clip_by_value_2/clip_by_value?
custom_model/tf.cast_2/CastCast5custom_model/tf.math.greater_equal_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
custom_model/tf.cast_2/Cast?
"custom_model/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"custom_model/tf.concat/concat/axis?
custom_model/tf.concat/concatConcatV22custom_model/model/tf.math.reduce_sum/Sum:output:06custom_model/model_1/tf.math.reduce_sum_1/Sum:output:0+custom_model/tf.concat/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
custom_model/tf.concat/concat?
$custom_model/tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$custom_model/tf.concat_1/concat/axis?
custom_model/tf.concat_1/concatConcatV21custom_model/tf.clip_by_value_2/clip_by_value:z:0custom_model/tf.cast_2/Cast:y:0-custom_model/tf.concat_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2!
custom_model/tf.concat_1/concat?
(custom_model/dense/MatMul/ReadVariableOpReadVariableOp1custom_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(custom_model/dense/MatMul/ReadVariableOp?
custom_model/dense/MatMulMatMul&custom_model/tf.concat/concat:output:00custom_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense/MatMul?
)custom_model/dense/BiasAdd/ReadVariableOpReadVariableOp2custom_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)custom_model/dense/BiasAdd/ReadVariableOp?
custom_model/dense/BiasAddBiasAdd#custom_model/dense/MatMul:product:01custom_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense/BiasAdd?
custom_model/dense/ReluRelu#custom_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model/dense/Relu?
*custom_model/dense_3/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*custom_model/dense_3/MatMul/ReadVariableOp?
custom_model/dense_3/MatMulMatMul(custom_model/tf.concat_1/concat:output:02custom_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_3/MatMul?
+custom_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_3/BiasAdd/ReadVariableOp?
custom_model/dense_3/BiasAddBiasAdd%custom_model/dense_3/MatMul:product:03custom_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_3/BiasAdd?
*custom_model/dense_1/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*custom_model/dense_1/MatMul/ReadVariableOp?
custom_model/dense_1/MatMulMatMul%custom_model/dense/Relu:activations:02custom_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_1/MatMul?
+custom_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_1/BiasAdd/ReadVariableOp?
custom_model/dense_1/BiasAddBiasAdd%custom_model/dense_1/MatMul:product:03custom_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_1/BiasAdd?
custom_model/dense_1/ReluRelu%custom_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_1/Relu?
*custom_model/dense_2/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*custom_model/dense_2/MatMul/ReadVariableOp?
custom_model/dense_2/MatMulMatMul'custom_model/dense_1/Relu:activations:02custom_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_2/MatMul?
+custom_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_2/BiasAdd/ReadVariableOp?
custom_model/dense_2/BiasAddBiasAdd%custom_model/dense_2/MatMul:product:03custom_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_2/BiasAdd?
custom_model/dense_2/ReluRelu%custom_model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_2/Relu?
*custom_model/dense_4/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*custom_model/dense_4/MatMul/ReadVariableOp?
custom_model/dense_4/MatMulMatMul%custom_model/dense_3/BiasAdd:output:02custom_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_4/MatMul?
+custom_model/dense_4/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_4/BiasAdd/ReadVariableOp?
custom_model/dense_4/BiasAddBiasAdd%custom_model/dense_4/MatMul:product:03custom_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_4/BiasAdd?
$custom_model/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$custom_model/tf.concat_2/concat/axis?
custom_model/tf.concat_2/concatConcatV2'custom_model/dense_2/Relu:activations:0%custom_model/dense_4/BiasAdd:output:0-custom_model/tf.concat_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2!
custom_model/tf.concat_2/concat?
*custom_model/dense_5/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*custom_model/dense_5/MatMul/ReadVariableOp?
custom_model/dense_5/MatMulMatMul(custom_model/tf.concat_2/concat:output:02custom_model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_5/MatMul?
+custom_model/dense_5/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_5/BiasAdd/ReadVariableOp?
custom_model/dense_5/BiasAddBiasAdd%custom_model/dense_5/MatMul:product:03custom_model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_5/BiasAdd?
custom_model/tf.nn.relu/ReluRelu%custom_model/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model/tf.nn.relu/Relu?
*custom_model/dense_6/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*custom_model/dense_6/MatMul/ReadVariableOp?
custom_model/dense_6/MatMulMatMul*custom_model/tf.nn.relu/Relu:activations:02custom_model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_6/MatMul?
+custom_model/dense_6/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_6/BiasAdd/ReadVariableOp?
custom_model/dense_6/BiasAddBiasAdd%custom_model/dense_6/MatMul:product:03custom_model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_6/BiasAdd?
)custom_model/tf.__operators__.add_4/AddV2AddV2%custom_model/dense_6/BiasAdd:output:0*custom_model/tf.nn.relu/Relu:activations:0*
T0*(
_output_shapes
:??????????2+
)custom_model/tf.__operators__.add_4/AddV2?
custom_model/tf.nn.relu_1/ReluRelu-custom_model/tf.__operators__.add_4/AddV2:z:0*
T0*(
_output_shapes
:??????????2 
custom_model/tf.nn.relu_1/Relu?
*custom_model/dense_7/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*custom_model/dense_7/MatMul/ReadVariableOp?
custom_model/dense_7/MatMulMatMul,custom_model/tf.nn.relu_1/Relu:activations:02custom_model/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_7/MatMul?
+custom_model/dense_7/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+custom_model/dense_7/BiasAdd/ReadVariableOp?
custom_model/dense_7/BiasAddBiasAdd%custom_model/dense_7/MatMul:product:03custom_model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
custom_model/dense_7/BiasAdd?
)custom_model/tf.__operators__.add_5/AddV2AddV2%custom_model/dense_7/BiasAdd:output:0,custom_model/tf.nn.relu_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2+
)custom_model/tf.__operators__.add_5/AddV2?
custom_model/tf.nn.relu_2/ReluRelu-custom_model/tf.__operators__.add_5/AddV2:z:0*
T0*(
_output_shapes
:??????????2 
custom_model/tf.nn.relu_2/Relu?
;custom_model/normalize/normalization/Reshape/ReadVariableOpReadVariableOpDcustom_model_normalize_normalization_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;custom_model/normalize/normalization/Reshape/ReadVariableOp?
2custom_model/normalize/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      24
2custom_model/normalize/normalization/Reshape/shape?
,custom_model/normalize/normalization/ReshapeReshapeCcustom_model/normalize/normalization/Reshape/ReadVariableOp:value:0;custom_model/normalize/normalization/Reshape/shape:output:0*
T0*
_output_shapes
:	?2.
,custom_model/normalize/normalization/Reshape?
=custom_model/normalize/normalization/Reshape_1/ReadVariableOpReadVariableOpFcustom_model_normalize_normalization_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02?
=custom_model/normalize/normalization/Reshape_1/ReadVariableOp?
4custom_model/normalize/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      26
4custom_model/normalize/normalization/Reshape_1/shape?
.custom_model/normalize/normalization/Reshape_1ReshapeEcustom_model/normalize/normalization/Reshape_1/ReadVariableOp:value:0=custom_model/normalize/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?20
.custom_model/normalize/normalization/Reshape_1?
(custom_model/normalize/normalization/subSub,custom_model/tf.nn.relu_2/Relu:activations:05custom_model/normalize/normalization/Reshape:output:0*
T0*(
_output_shapes
:??????????2*
(custom_model/normalize/normalization/sub?
)custom_model/normalize/normalization/SqrtSqrt7custom_model/normalize/normalization/Reshape_1:output:0*
T0*
_output_shapes
:	?2+
)custom_model/normalize/normalization/Sqrt?
.custom_model/normalize/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???320
.custom_model/normalize/normalization/Maximum/y?
,custom_model/normalize/normalization/MaximumMaximum-custom_model/normalize/normalization/Sqrt:y:07custom_model/normalize/normalization/Maximum/y:output:0*
T0*
_output_shapes
:	?2.
,custom_model/normalize/normalization/Maximum?
,custom_model/normalize/normalization/truedivRealDiv,custom_model/normalize/normalization/sub:z:00custom_model/normalize/normalization/Maximum:z:0*
T0*(
_output_shapes
:??????????2.
,custom_model/normalize/normalization/truediv?
*custom_model/dense_8/MatMul/ReadVariableOpReadVariableOp3custom_model_dense_8_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*custom_model/dense_8/MatMul/ReadVariableOp?
custom_model/dense_8/MatMulMatMul0custom_model/normalize/normalization/truediv:z:02custom_model/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
custom_model/dense_8/MatMul?
+custom_model/dense_8/BiasAdd/ReadVariableOpReadVariableOp4custom_model_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+custom_model/dense_8/BiasAdd/ReadVariableOp?
custom_model/dense_8/BiasAddBiasAdd%custom_model/dense_8/MatMul:product:03custom_model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
custom_model/dense_8/BiasAdd?

IdentityIdentity%custom_model/dense_8/BiasAdd:output:0*^custom_model/dense/BiasAdd/ReadVariableOp)^custom_model/dense/MatMul/ReadVariableOp,^custom_model/dense_1/BiasAdd/ReadVariableOp+^custom_model/dense_1/MatMul/ReadVariableOp,^custom_model/dense_2/BiasAdd/ReadVariableOp+^custom_model/dense_2/MatMul/ReadVariableOp,^custom_model/dense_3/BiasAdd/ReadVariableOp+^custom_model/dense_3/MatMul/ReadVariableOp,^custom_model/dense_4/BiasAdd/ReadVariableOp+^custom_model/dense_4/MatMul/ReadVariableOp,^custom_model/dense_5/BiasAdd/ReadVariableOp+^custom_model/dense_5/MatMul/ReadVariableOp,^custom_model/dense_6/BiasAdd/ReadVariableOp+^custom_model/dense_6/MatMul/ReadVariableOp,^custom_model/dense_7/BiasAdd/ReadVariableOp+^custom_model/dense_7/MatMul/ReadVariableOp,^custom_model/dense_8/BiasAdd/ReadVariableOp+^custom_model/dense_8/MatMul/ReadVariableOp.^custom_model/model/embedding/embedding_lookup0^custom_model/model/embedding_1/embedding_lookup0^custom_model/model/embedding_2/embedding_lookup2^custom_model/model_1/embedding_3/embedding_lookup2^custom_model/model_1/embedding_4/embedding_lookup2^custom_model/model_1/embedding_5/embedding_lookup<^custom_model/normalize/normalization/Reshape/ReadVariableOp>^custom_model/normalize/normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2V
)custom_model/dense/BiasAdd/ReadVariableOp)custom_model/dense/BiasAdd/ReadVariableOp2T
(custom_model/dense/MatMul/ReadVariableOp(custom_model/dense/MatMul/ReadVariableOp2Z
+custom_model/dense_1/BiasAdd/ReadVariableOp+custom_model/dense_1/BiasAdd/ReadVariableOp2X
*custom_model/dense_1/MatMul/ReadVariableOp*custom_model/dense_1/MatMul/ReadVariableOp2Z
+custom_model/dense_2/BiasAdd/ReadVariableOp+custom_model/dense_2/BiasAdd/ReadVariableOp2X
*custom_model/dense_2/MatMul/ReadVariableOp*custom_model/dense_2/MatMul/ReadVariableOp2Z
+custom_model/dense_3/BiasAdd/ReadVariableOp+custom_model/dense_3/BiasAdd/ReadVariableOp2X
*custom_model/dense_3/MatMul/ReadVariableOp*custom_model/dense_3/MatMul/ReadVariableOp2Z
+custom_model/dense_4/BiasAdd/ReadVariableOp+custom_model/dense_4/BiasAdd/ReadVariableOp2X
*custom_model/dense_4/MatMul/ReadVariableOp*custom_model/dense_4/MatMul/ReadVariableOp2Z
+custom_model/dense_5/BiasAdd/ReadVariableOp+custom_model/dense_5/BiasAdd/ReadVariableOp2X
*custom_model/dense_5/MatMul/ReadVariableOp*custom_model/dense_5/MatMul/ReadVariableOp2Z
+custom_model/dense_6/BiasAdd/ReadVariableOp+custom_model/dense_6/BiasAdd/ReadVariableOp2X
*custom_model/dense_6/MatMul/ReadVariableOp*custom_model/dense_6/MatMul/ReadVariableOp2Z
+custom_model/dense_7/BiasAdd/ReadVariableOp+custom_model/dense_7/BiasAdd/ReadVariableOp2X
*custom_model/dense_7/MatMul/ReadVariableOp*custom_model/dense_7/MatMul/ReadVariableOp2Z
+custom_model/dense_8/BiasAdd/ReadVariableOp+custom_model/dense_8/BiasAdd/ReadVariableOp2X
*custom_model/dense_8/MatMul/ReadVariableOp*custom_model/dense_8/MatMul/ReadVariableOp2^
-custom_model/model/embedding/embedding_lookup-custom_model/model/embedding/embedding_lookup2b
/custom_model/model/embedding_1/embedding_lookup/custom_model/model/embedding_1/embedding_lookup2b
/custom_model/model/embedding_2/embedding_lookup/custom_model/model/embedding_2/embedding_lookup2f
1custom_model/model_1/embedding_3/embedding_lookup1custom_model/model_1/embedding_3/embedding_lookup2f
1custom_model/model_1/embedding_4/embedding_lookup1custom_model/model_1/embedding_4/embedding_lookup2f
1custom_model/model_1/embedding_5/embedding_lookup1custom_model/model_1/embedding_5/embedding_lookup2z
;custom_model/normalize/normalization/Reshape/ReadVariableOp;custom_model/normalize/normalization/Reshape/ReadVariableOp2~
=custom_model/normalize/normalization/Reshape_1/ReadVariableOp=custom_model/normalize/normalization/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
E__inference_embedding_4_layer_call_and_return_conditional_losses_1746

inputs
embedding_lookup_1740
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_1740Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*(
_class
loc:@embedding_lookup/1740*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1740*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_3314

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_16062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
&__inference_model_1_layer_call_fn_3437

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_18772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
bets-
serving_default_bets:0?????????

9
cards0/
serving_default_cards0:0?????????
9
cards1/
serving_default_cards1:0?????????;
dense_80
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "CustomModel", "name": "custom_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value", "inbound_nodes": [["flatten", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["tf.clip_by_value", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["tf.compat.v1.floor_div", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal", "inbound_nodes": [["flatten", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["embedding_2", 0, 0, {"y": ["embedding", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["tf.math.floormod", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast", "inbound_nodes": [["tf.math.greater_equal", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["embedding_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims", "inbound_nodes": [["tf.cast", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.expand_dims", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.multiply", 0, 0, {"axis": 1}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.math.reduce_sum", 0, 0]]}, "name": "model", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_1", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_1", "inbound_nodes": [["flatten_1", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_1", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_5", "inbound_nodes": [[["tf.clip_by_value_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_3", "inbound_nodes": [[["tf.compat.v1.floor_div_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_1", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_1", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_1", "inbound_nodes": [["flatten_1", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["embedding_5", 0, 0, {"y": ["embedding_3", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_4", "inbound_nodes": [[["tf.math.floormod_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_1", "inbound_nodes": [["tf.math.greater_equal_1", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {"y": ["embedding_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_1", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_1", "inbound_nodes": [["tf.cast_1", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.expand_dims_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"axis": 1}]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["tf.math.reduce_sum_1", 0, 0]]}, "name": "model_1", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_2", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_2", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["model", 1, 0, {"axis": 1}], ["model_1", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_2", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_2", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_2", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_2", "inbound_nodes": [["tf.math.greater_equal_2", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["tf.concat", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_1", "inbound_nodes": [[["tf.clip_by_value_2", 0, 0, {"axis": -1}], ["tf.cast_2", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["tf.concat_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_2", "inbound_nodes": [[["dense_2", 0, 0, {"axis": -1}], ["dense_4", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["tf.concat_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu", "inbound_nodes": [["dense_5", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["tf.nn.relu", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["dense_6", 0, 0, {"y": ["tf.nn.relu", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_1", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_1", "inbound_nodes": [["tf.__operators__.add_4", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["tf.nn.relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["dense_7", 0, 0, {"y": ["tf.nn.relu_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_2", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_2", "inbound_nodes": [["tf.__operators__.add_5", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize", "trainable": true, "dtype": "float32"}, "name": "normalize", "inbound_nodes": [[["tf.nn.relu_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -5}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["normalize", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_8", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value", "inbound_nodes": [["flatten", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["tf.clip_by_value", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["tf.compat.v1.floor_div", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal", "inbound_nodes": [["flatten", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["embedding_2", 0, 0, {"y": ["embedding", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["tf.math.floormod", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast", "inbound_nodes": [["tf.math.greater_equal", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["embedding_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims", "inbound_nodes": [["tf.cast", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.expand_dims", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.multiply", 0, 0, {"axis": 1}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.math.reduce_sum", 0, 0]]}, "name": "model", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_1", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_1", "inbound_nodes": [["flatten_1", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_1", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_5", "inbound_nodes": [[["tf.clip_by_value_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_3", "inbound_nodes": [[["tf.compat.v1.floor_div_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_1", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_1", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_1", "inbound_nodes": [["flatten_1", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["embedding_5", 0, 0, {"y": ["embedding_3", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_4", "inbound_nodes": [[["tf.math.floormod_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_1", "inbound_nodes": [["tf.math.greater_equal_1", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {"y": ["embedding_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_1", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_1", "inbound_nodes": [["tf.cast_1", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.expand_dims_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"axis": 1}]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["tf.math.reduce_sum_1", 0, 0]]}, "name": "model_1", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_2", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_2", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat", "inbound_nodes": [[["model", 1, 0, {"axis": 1}], ["model_1", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_2", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_2", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_2", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_2", "inbound_nodes": [["tf.math.greater_equal_2", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["tf.concat", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_1", "inbound_nodes": [[["tf.clip_by_value_2", 0, 0, {"axis": -1}], ["tf.cast_2", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["tf.concat_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_2", "inbound_nodes": [[["dense_2", 0, 0, {"axis": -1}], ["dense_4", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["tf.concat_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu", "inbound_nodes": [["dense_5", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["tf.nn.relu", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["dense_6", 0, 0, {"y": ["tf.nn.relu", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_1", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_1", "inbound_nodes": [["tf.__operators__.add_4", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["tf.nn.relu_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["dense_7", 0, 0, {"y": ["tf.nn.relu_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_2", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_2", "inbound_nodes": [["tf.__operators__.add_5", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize", "trainable": true, "dtype": "float32"}, "name": "normalize", "inbound_nodes": [[["tf.nn.relu_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -5}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["normalize", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_8", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?P
 layer-0
!layer-1
"layer-2
#layer-3
$layer_with_weights-0
$layer-4
%layer_with_weights-1
%layer-5
&layer-6
'layer-7
(layer-8
)layer_with_weights-2
)layer-9
*layer-10
+layer-11
,layer-12
-layer-13
.layer-14
/	variables
0regularization_losses
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?L
_tf_keras_network?L{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value", "inbound_nodes": [["flatten", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["tf.clip_by_value", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["tf.compat.v1.floor_div", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal", "inbound_nodes": [["flatten", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["embedding_2", 0, 0, {"y": ["embedding", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["tf.math.floormod", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast", "inbound_nodes": [["tf.math.greater_equal", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["embedding_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims", "inbound_nodes": [["tf.cast", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.expand_dims", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.multiply", 0, 0, {"axis": 1}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.math.reduce_sum", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value", "inbound_nodes": [["flatten", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_2", "inbound_nodes": [[["tf.clip_by_value", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["tf.compat.v1.floor_div", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod", "inbound_nodes": [["tf.clip_by_value", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal", "inbound_nodes": [["flatten", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["embedding_2", 0, 0, {"y": ["embedding", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["tf.math.floormod", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast", "inbound_nodes": [["tf.math.greater_equal", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["tf.__operators__.add", 0, 0, {"y": ["embedding_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims", "inbound_nodes": [["tf.cast", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {"y": ["tf.expand_dims", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.multiply", 0, 0, {"axis": 1}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.math.reduce_sum", 0, 0]]}}}
?Q
3layer-0
4layer-1
5layer-2
6layer-3
7layer_with_weights-0
7layer-4
8layer_with_weights-1
8layer-5
9layer-6
:layer-7
;layer-8
<layer_with_weights-2
<layer-9
=layer-10
>layer-11
?layer-12
@layer-13
Alayer-14
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?N
_tf_keras_network?M{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_1", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_1", "inbound_nodes": [["flatten_1", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_1", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_5", "inbound_nodes": [[["tf.clip_by_value_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_3", "inbound_nodes": [[["tf.compat.v1.floor_div_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_1", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_1", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_1", "inbound_nodes": [["flatten_1", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["embedding_5", 0, 0, {"y": ["embedding_3", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_4", "inbound_nodes": [[["tf.math.floormod_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_1", "inbound_nodes": [["tf.math.greater_equal_1", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {"y": ["embedding_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_1", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_1", "inbound_nodes": [["tf.cast_1", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.expand_dims_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"axis": 1}]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["tf.math.reduce_sum_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_1", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_1", "inbound_nodes": [["flatten_1", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_1", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_5", "inbound_nodes": [[["tf.clip_by_value_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_3", "inbound_nodes": [[["tf.compat.v1.floor_div_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_1", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_1", "inbound_nodes": [["tf.clip_by_value_1", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_1", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_1", "inbound_nodes": [["flatten_1", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["embedding_5", 0, 0, {"y": ["embedding_3", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_4", "inbound_nodes": [[["tf.math.floormod_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_1", "inbound_nodes": [["tf.math.greater_equal_1", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {"y": ["embedding_4", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_1", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_1", "inbound_nodes": [["tf.cast_1", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {"y": ["tf.expand_dims_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"axis": 1}]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["tf.math.reduce_sum_1", 0, 0]]}}}
?
F	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_2", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
G	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat", "trainable": true, "dtype": "float32", "function": "concat"}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_2", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_2", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
P	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_1", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Qkernel
Rbias
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

]kernel
^bias
_	variables
`regularization_losses
atrainable_variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ckernel
dbias
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
i	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_2", "trainable": true, "dtype": "float32", "function": "concat"}}
?

jkernel
kbias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
p	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

qkernel
rbias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
w	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
x	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_1", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

ykernel
zbias
{	variables
|regularization_losses
}trainable_variables
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_2", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -5}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?0
?1
?2
?3
?4
?5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
?22
?23
?24
?25
?26"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
J6
K7
Q8
R9
W10
X11
]12
^13
c14
d15
j16
k17
q18
r19
y20
z21
?22
?23"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
	variables
?metrics
regularization_losses
?layer_metrics
?non_trainable_variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
/	variables
?metrics
0regularization_losses
?layer_metrics
?non_trainable_variables
1trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_1", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_1", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_1", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_1", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_1", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_1", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_1", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
B	variables
?metrics
Cregularization_losses
?layer_metrics
?non_trainable_variables
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 :
??2dense/kernel
:?2
dense/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
L	variables
?metrics
Mregularization_losses
?layer_metrics
?non_trainable_variables
Ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
": 
??2dense_1/kernel
:?2dense_1/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
S	variables
?metrics
Tregularization_losses
?layer_metrics
?non_trainable_variables
Utrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_3/kernel
:?2dense_3/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
Y	variables
?metrics
Zregularization_losses
?layer_metrics
?non_trainable_variables
[trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
_	variables
?metrics
`regularization_losses
?layer_metrics
?non_trainable_variables
atrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_4/kernel
:?2dense_4/bias
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
e	variables
?metrics
fregularization_losses
?layer_metrics
?non_trainable_variables
gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
": 
??2dense_5/kernel
:?2dense_5/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
l	variables
?metrics
mregularization_losses
?layer_metrics
?non_trainable_variables
ntrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
": 
??2dense_6/kernel
:?2dense_6/bias
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
s	variables
?metrics
tregularization_losses
?layer_metrics
?non_trainable_variables
utrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
": 
??2dense_7/kernel
:?2dense_7/bias
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
{	variables
?metrics
|regularization_losses
?layer_metrics
?non_trainable_variables
}trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_8/kernel
:2dense_8/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'	4?2embedding_2/embeddings
':%	?2embedding/embeddings
):'	?2embedding_1/embeddings
):'	4?2embedding_5/embeddings
):'	?2embedding_3/embeddings
):'	?2embedding_4/embeddings
):'?2normalize/normalization/mean
-:+?2 normalize/normalization/variance
%:#	 2normalize/normalization/count
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
?
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
?
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
?12
@13
A14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
F
	?mean
?variance

?count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
__inference__wrapped_model_1436?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

?2?
F__inference_custom_model_layer_call_and_return_conditional_losses_2253
F__inference_custom_model_layer_call_and_return_conditional_losses_3079
F__inference_custom_model_layer_call_and_return_conditional_losses_2909
F__inference_custom_model_layer_call_and_return_conditional_losses_2345?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_custom_model_layer_call_fn_3148
+__inference_custom_model_layer_call_fn_2668
+__inference_custom_model_layer_call_fn_2507
+__inference_custom_model_layer_call_fn_3217?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_model_layer_call_and_return_conditional_losses_3301
?__inference_model_layer_call_and_return_conditional_losses_1539
?__inference_model_layer_call_and_return_conditional_losses_1571
?__inference_model_layer_call_and_return_conditional_losses_3259?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_3314
$__inference_model_layer_call_fn_1617
$__inference_model_layer_call_fn_3327
$__inference_model_layer_call_fn_1662?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_1_layer_call_and_return_conditional_losses_3369
A__inference_model_1_layer_call_and_return_conditional_losses_1797
A__inference_model_1_layer_call_and_return_conditional_losses_1765
A__inference_model_1_layer_call_and_return_conditional_losses_3411?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_model_1_layer_call_fn_1888
&__inference_model_1_layer_call_fn_3424
&__inference_model_1_layer_call_fn_3437
&__inference_model_1_layer_call_fn_1843?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_3448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_3457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_3468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_3477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_3_layer_call_and_return_conditional_losses_3487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_3_layer_call_fn_3496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_3507?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_3516?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_4_layer_call_and_return_conditional_losses_3526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_4_layer_call_fn_3535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_5_layer_call_and_return_conditional_losses_3545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_5_layer_call_fn_3554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_6_layer_call_and_return_conditional_losses_3564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_6_layer_call_fn_3573?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_7_layer_call_and_return_conditional_losses_3583?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_7_layer_call_fn_3592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_normalize_layer_call_and_return_conditional_losses_3609?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_normalize_layer_call_fn_3618?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_8_layer_call_and_return_conditional_losses_3628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_8_layer_call_fn_3637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_2739betscards0cards1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_3643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_flatten_layer_call_fn_3648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_2_layer_call_and_return_conditional_losses_3658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_2_layer_call_fn_3665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_embedding_layer_call_and_return_conditional_losses_3675?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_embedding_layer_call_fn_3682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_1_layer_call_and_return_conditional_losses_3692?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_1_layer_call_fn_3699?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_1_layer_call_and_return_conditional_losses_3705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_flatten_1_layer_call_fn_3710?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_5_layer_call_and_return_conditional_losses_3720?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_5_layer_call_fn_3727?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_3_layer_call_and_return_conditional_losses_3737?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_3_layer_call_fn_3744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_4_layer_call_and_return_conditional_losses_3754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_4_layer_call_fn_3761?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4?
__inference__wrapped_model_1436?.???????????JKWXQR]^cdjkqryz????{?x
q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

? "1?.
,
dense_8!?
dense_8??????????
F__inference_custom_model_layer_call_and_return_conditional_losses_2253?.???????????JKWXQR]^cdjkqryz???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "%?"
?
0?????????
? ?
F__inference_custom_model_layer_call_and_return_conditional_losses_2345?.???????????JKWXQR]^cdjkqryz???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "%?"
?
0?????????
? ?
F__inference_custom_model_layer_call_and_return_conditional_losses_2909?.???????????JKWXQR]^cdjkqryz???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "%?"
?
0?????????
? ?
F__inference_custom_model_layer_call_and_return_conditional_losses_3079?.???????????JKWXQR]^cdjkqryz???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "%?"
?
0?????????
? ?
+__inference_custom_model_layer_call_fn_2507?.???????????JKWXQR]^cdjkqryz???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "???????????
+__inference_custom_model_layer_call_fn_2668?.???????????JKWXQR]^cdjkqryz???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "???????????
+__inference_custom_model_layer_call_fn_3148?.???????????JKWXQR]^cdjkqryz???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "???????????
+__inference_custom_model_layer_call_fn_3217?.???????????JKWXQR]^cdjkqryz???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "???????????
A__inference_dense_1_layer_call_and_return_conditional_losses_3468^QR0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_1_layer_call_fn_3477QQR0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_2_layer_call_and_return_conditional_losses_3507^]^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_2_layer_call_fn_3516Q]^0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_3_layer_call_and_return_conditional_losses_3487]WX/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? z
&__inference_dense_3_layer_call_fn_3496PWX/?,
%?"
 ?
inputs?????????
? "????????????
A__inference_dense_4_layer_call_and_return_conditional_losses_3526^cd0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_4_layer_call_fn_3535Qcd0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_5_layer_call_and_return_conditional_losses_3545^jk0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_5_layer_call_fn_3554Qjk0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_6_layer_call_and_return_conditional_losses_3564^qr0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_6_layer_call_fn_3573Qqr0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_7_layer_call_and_return_conditional_losses_3583^yz0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_7_layer_call_fn_3592Qyz0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_8_layer_call_and_return_conditional_losses_3628_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
&__inference_dense_8_layer_call_fn_3637R??0?-
&?#
!?
inputs??????????
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_3448^JK0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? y
$__inference_dense_layer_call_fn_3457QJK0?-
&?#
!?
inputs??????????
? "????????????
E__inference_embedding_1_layer_call_and_return_conditional_losses_3692a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
*__inference_embedding_1_layer_call_fn_3699T?/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_embedding_2_layer_call_and_return_conditional_losses_3658a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
*__inference_embedding_2_layer_call_fn_3665T?/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_embedding_3_layer_call_and_return_conditional_losses_3737a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
*__inference_embedding_3_layer_call_fn_3744T?/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_embedding_4_layer_call_and_return_conditional_losses_3754a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
*__inference_embedding_4_layer_call_fn_3761T?/?,
%?"
 ?
inputs?????????
? "????????????
E__inference_embedding_5_layer_call_and_return_conditional_losses_3720a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
*__inference_embedding_5_layer_call_fn_3727T?/?,
%?"
 ?
inputs?????????
? "????????????
C__inference_embedding_layer_call_and_return_conditional_losses_3675a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
(__inference_embedding_layer_call_fn_3682T?/?,
%?"
 ?
inputs?????????
? "????????????
C__inference_flatten_1_layer_call_and_return_conditional_losses_3705X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? w
(__inference_flatten_1_layer_call_fn_3710K/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_flatten_layer_call_and_return_conditional_losses_3643X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? u
&__inference_flatten_layer_call_fn_3648K/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_model_1_layer_call_and_return_conditional_losses_1765l????8?5
.?+
!?
input_2?????????
p

 
? "&?#
?
0??????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_1797l????8?5
.?+
!?
input_2?????????
p 

 
? "&?#
?
0??????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_3369k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_3411k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
&__inference_model_1_layer_call_fn_1843_????8?5
.?+
!?
input_2?????????
p

 
? "????????????
&__inference_model_1_layer_call_fn_1888_????8?5
.?+
!?
input_2?????????
p 

 
? "????????????
&__inference_model_1_layer_call_fn_3424^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
&__inference_model_1_layer_call_fn_3437^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
?__inference_model_layer_call_and_return_conditional_losses_1539l????8?5
.?+
!?
input_1?????????
p

 
? "&?#
?
0??????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_1571l????8?5
.?+
!?
input_1?????????
p 

 
? "&?#
?
0??????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3259k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_3301k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
$__inference_model_layer_call_fn_1617_????8?5
.?+
!?
input_1?????????
p

 
? "????????????
$__inference_model_layer_call_fn_1662_????8?5
.?+
!?
input_1?????????
p 

 
? "????????????
$__inference_model_layer_call_fn_3314^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
$__inference_model_layer_call_fn_3327^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
C__inference_normalize_layer_call_and_return_conditional_losses_3609[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? z
(__inference_normalize_layer_call_fn_3618N??+?(
!?
?
x??????????
? "????????????
"__inference_signature_wrapper_2739?.???????????JKWXQR]^cdjkqryz???????
? 
???
&
bets?
bets?????????

*
cards0 ?
cards0?????????
*
cards1 ?
cards1?????????"1?.
,
dense_8!?
dense_8?????????