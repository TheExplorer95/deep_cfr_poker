??!
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
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
|
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_36/kernel
u
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel* 
_output_shapes
:
??*
dtype0
s
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_36/bias
l
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes	
:?*
dtype0
|
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_37/kernel
u
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel* 
_output_shapes
:
??*
dtype0
s
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_37/bias
l
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes	
:?*
dtype0
{
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_39/kernel
t
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes
:	?*
dtype0
s
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_39/bias
l
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes	
:?*
dtype0
|
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_38/kernel
u
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel* 
_output_shapes
:
??*
dtype0
s
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_38/bias
l
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes	
:?*
dtype0
|
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_40/kernel
u
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel* 
_output_shapes
:
??*
dtype0
s
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_40/bias
l
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes	
:?*
dtype0
|
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_41/kernel
u
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel* 
_output_shapes
:
??*
dtype0
s
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_41/bias
l
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
_output_shapes	
:?*
dtype0
|
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_42/kernel
u
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel* 
_output_shapes
:
??*
dtype0
s
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_42/bias
l
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes	
:?*
dtype0
|
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_43/kernel
u
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel* 
_output_shapes
:
??*
dtype0
s
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_43/bias
l
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes	
:?*
dtype0
{
dense_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_44/kernel
t
#dense_44/kernel/Read/ReadVariableOpReadVariableOpdense_44/kernel*
_output_shapes
:	?*
dtype0
r
dense_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_44/bias
k
!dense_44/bias/Read/ReadVariableOpReadVariableOpdense_44/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
embedding_26/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_26/embeddings
?
+embedding_26/embeddings/Read/ReadVariableOpReadVariableOpembedding_26/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_24/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_24/embeddings
?
+embedding_24/embeddings/Read/ReadVariableOpReadVariableOpembedding_24/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_25/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_25/embeddings
?
+embedding_25/embeddings/Read/ReadVariableOpReadVariableOpembedding_25/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_29/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_29/embeddings
?
+embedding_29/embeddings/Read/ReadVariableOpReadVariableOpembedding_29/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_27/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_27/embeddings
?
+embedding_27/embeddings/Read/ReadVariableOpReadVariableOpembedding_27/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_28/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_28/embeddings
?
+embedding_28/embeddings/Read/ReadVariableOpReadVariableOpembedding_28/embeddings*
_output_shapes
:	?*
dtype0
?
 normalize_4/normalization_4/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" normalize_4/normalization_4/mean
?
4normalize_4/normalization_4/mean/Read/ReadVariableOpReadVariableOp normalize_4/normalization_4/mean*
_output_shapes	
:?*
dtype0
?
$normalize_4/normalization_4/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$normalize_4/normalization_4/variance
?
8normalize_4/normalization_4/variance/Read/ReadVariableOpReadVariableOp$normalize_4/normalization_4/variance*
_output_shapes	
:?*
dtype0
?
!normalize_4/normalization_4/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_4/normalization_4/count
?
5normalize_4/normalization_4/count/Read/ReadVariableOpReadVariableOp!normalize_4/normalization_4/count*
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
?
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_36/bias/m
z
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_37/bias/m
z
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_39/kernel/m
?
*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_39/bias/m
z
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_38/kernel/m
?
*Adam/dense_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_38/bias/m
z
(Adam/dense_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_40/kernel/m
?
*Adam/dense_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_40/bias/m
z
(Adam/dense_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_41/kernel/m
?
*Adam/dense_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_41/bias/m
z
(Adam/dense_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_42/kernel/m
?
*Adam/dense_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_42/bias/m
z
(Adam/dense_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_43/kernel/m
?
*Adam/dense_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_43/bias/m
z
(Adam/dense_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_44/kernel/m
?
*Adam/dense_44/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_44/bias/m
y
(Adam/dense_44/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_26/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_26/embeddings/m
?
2Adam/embedding_26/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_26/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_24/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_24/embeddings/m
?
2Adam/embedding_24/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_24/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_25/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_25/embeddings/m
?
2Adam/embedding_25/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_25/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_29/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_29/embeddings/m
?
2Adam/embedding_29/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_29/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_27/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_27/embeddings/m
?
2Adam/embedding_27/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_27/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_28/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_28/embeddings/m
?
2Adam/embedding_28/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_28/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_36/bias/v
z
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_37/bias/v
z
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_39/kernel/v
?
*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_39/bias/v
z
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_38/kernel/v
?
*Adam/dense_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_38/bias/v
z
(Adam/dense_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_40/kernel/v
?
*Adam/dense_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_40/bias/v
z
(Adam/dense_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_41/kernel/v
?
*Adam/dense_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_41/bias/v
z
(Adam/dense_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_42/kernel/v
?
*Adam/dense_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_42/bias/v
z
(Adam/dense_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_43/kernel/v
?
*Adam/dense_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_43/bias/v
z
(Adam/dense_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_44/kernel/v
?
*Adam/dense_44/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_44/bias/v
y
(Adam/dense_44/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_44/bias/v*
_output_shapes
:*
dtype0
?
Adam/embedding_26/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_26/embeddings/v
?
2Adam/embedding_26/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_26/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_24/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_24/embeddings/v
?
2Adam/embedding_24/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_24/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_25/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_25/embeddings/v
?
2Adam/embedding_25/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_25/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_29/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_29/embeddings/v
?
2Adam/embedding_29/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_29/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_27/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_27/embeddings/v
?
2Adam/embedding_27/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_27/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_28/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_28/embeddings/v
?
2Adam/embedding_28/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_28/embeddings/v*
_output_shapes
:	?*
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
??
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
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
	optimizer
loss
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
 
 
 
?
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api
h

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api

R	keras_api
h

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
h

Ykernel
Zbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
h

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api

k	keras_api
h

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api

r	keras_api
h

skernel
tbias
uregularization_losses
v	variables
wtrainable_variables
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
?	keras_api

?	keras_api

?	keras_api
f
?	normalize
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
 
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
regularization_losses
	variables
?metrics
trainable_variables
 
 
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
g
?
embeddings
?regularization_losses
?	variables
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
?regularization_losses
?	variables
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
 

?0
?1
?2

?0
?1
?2
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
1regularization_losses
2	variables
?metrics
3trainable_variables
 
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
g
?
embeddings
?regularization_losses
?	variables
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
?regularization_losses
?	variables
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
 

?0
?1
?2

?0
?1
?2
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
Dregularization_losses
E	variables
?metrics
Ftrainable_variables
 
 
 
 
[Y
VARIABLE_VALUEdense_36/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_36/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
Nregularization_losses
O	variables
?metrics
Ptrainable_variables
 
[Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
Uregularization_losses
V	variables
?metrics
Wtrainable_variables
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Y0
Z1

Y0
Z1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
[regularization_losses
\	variables
?metrics
]trainable_variables
[Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_38/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

_0
`1

_0
`1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
aregularization_losses
b	variables
?metrics
ctrainable_variables
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
gregularization_losses
h	variables
?metrics
itrainable_variables
 
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

l0
m1

l0
m1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
nregularization_losses
o	variables
?metrics
ptrainable_variables
 
[Y
VARIABLE_VALUEdense_42/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_42/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

s0
t1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
uregularization_losses
v	variables
?metrics
wtrainable_variables
 
 
[Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_43/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

{0
|1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
}regularization_losses
~	variables
?metrics
trainable_variables
 
 
c
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api
 

?0
?1
?2
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
\Z
VARIABLE_VALUEdense_44/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_44/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_26/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_24/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_25/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_29/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_27/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_28/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_4/normalization_4/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_4/normalization_4/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_4/normalization_4/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
?2
 
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
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 
 
 

?0

?0
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 

?0

?0
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 
 
 
 

?0

?0
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 
 
 
 
 
 
 
 
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
 
 
 
 
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 
 
 

?0

?0
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 

?0

?0
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 
 
 
 

?0

?0
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
 
 
 
 
 
 
 
 
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
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

?0
?1
?2
 
 

?0
 
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
~|
VARIABLE_VALUEAdam/dense_36/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_38/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_38/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_40/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_40/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_41/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_41/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_42/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_42/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_43/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_43/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_44/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_44/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_26/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_24/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_25/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_29/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_27/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_28/embeddings/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_36/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_38/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_38/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_40/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_40/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_41/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_41/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_42/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_42/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_43/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_43/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_44/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_44/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_26/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_24/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_25/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_29/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_27/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_28/embeddings/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_betsPlaceholder*'
_output_shapes
:?????????*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_26/embeddingsembedding_24/embeddingsembedding_25/embeddingsConst_2embedding_29/embeddingsembedding_27/embeddingsembedding_28/embeddingsConst_3Const_4dense_36/kerneldense_36/biasdense_39/kerneldense_39/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/bias normalize_4/normalization_4/mean$normalize_4/normalization_4/variancedense_44/kerneldense_44/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *0
f+R)
'__inference_signature_wrapper_300119886
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOp#dense_42/kernel/Read/ReadVariableOp!dense_42/bias/Read/ReadVariableOp#dense_43/kernel/Read/ReadVariableOp!dense_43/bias/Read/ReadVariableOp#dense_44/kernel/Read/ReadVariableOp!dense_44/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+embedding_26/embeddings/Read/ReadVariableOp+embedding_24/embeddings/Read/ReadVariableOp+embedding_25/embeddings/Read/ReadVariableOp+embedding_29/embeddings/Read/ReadVariableOp+embedding_27/embeddings/Read/ReadVariableOp+embedding_28/embeddings/Read/ReadVariableOp4normalize_4/normalization_4/mean/Read/ReadVariableOp8normalize_4/normalization_4/variance/Read/ReadVariableOp5normalize_4/normalization_4/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOp*Adam/dense_38/kernel/m/Read/ReadVariableOp(Adam/dense_38/bias/m/Read/ReadVariableOp*Adam/dense_40/kernel/m/Read/ReadVariableOp(Adam/dense_40/bias/m/Read/ReadVariableOp*Adam/dense_41/kernel/m/Read/ReadVariableOp(Adam/dense_41/bias/m/Read/ReadVariableOp*Adam/dense_42/kernel/m/Read/ReadVariableOp(Adam/dense_42/bias/m/Read/ReadVariableOp*Adam/dense_43/kernel/m/Read/ReadVariableOp(Adam/dense_43/bias/m/Read/ReadVariableOp*Adam/dense_44/kernel/m/Read/ReadVariableOp(Adam/dense_44/bias/m/Read/ReadVariableOp2Adam/embedding_26/embeddings/m/Read/ReadVariableOp2Adam/embedding_24/embeddings/m/Read/ReadVariableOp2Adam/embedding_25/embeddings/m/Read/ReadVariableOp2Adam/embedding_29/embeddings/m/Read/ReadVariableOp2Adam/embedding_27/embeddings/m/Read/ReadVariableOp2Adam/embedding_28/embeddings/m/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOp*Adam/dense_38/kernel/v/Read/ReadVariableOp(Adam/dense_38/bias/v/Read/ReadVariableOp*Adam/dense_40/kernel/v/Read/ReadVariableOp(Adam/dense_40/bias/v/Read/ReadVariableOp*Adam/dense_41/kernel/v/Read/ReadVariableOp(Adam/dense_41/bias/v/Read/ReadVariableOp*Adam/dense_42/kernel/v/Read/ReadVariableOp(Adam/dense_42/bias/v/Read/ReadVariableOp*Adam/dense_43/kernel/v/Read/ReadVariableOp(Adam/dense_43/bias/v/Read/ReadVariableOp*Adam/dense_44/kernel/v/Read/ReadVariableOp(Adam/dense_44/bias/v/Read/ReadVariableOp2Adam/embedding_26/embeddings/v/Read/ReadVariableOp2Adam/embedding_24/embeddings/v/Read/ReadVariableOp2Adam/embedding_25/embeddings/v/Read/ReadVariableOp2Adam/embedding_29/embeddings/v/Read/ReadVariableOp2Adam/embedding_27/embeddings/v/Read/ReadVariableOp2Adam/embedding_28/embeddings/v/Read/ReadVariableOpConst_5*_
TinX
V2T		*
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
GPU2 *0J 8? *+
f&R$
"__inference__traced_save_300121184
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_39/kerneldense_39/biasdense_38/kerneldense_38/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/biasdense_44/kerneldense_44/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_26/embeddingsembedding_24/embeddingsembedding_25/embeddingsembedding_29/embeddingsembedding_27/embeddingsembedding_28/embeddings normalize_4/normalization_4/mean$normalize_4/normalization_4/variance!normalize_4/normalization_4/counttotalcountAdam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/dense_39/kernel/mAdam/dense_39/bias/mAdam/dense_38/kernel/mAdam/dense_38/bias/mAdam/dense_40/kernel/mAdam/dense_40/bias/mAdam/dense_41/kernel/mAdam/dense_41/bias/mAdam/dense_42/kernel/mAdam/dense_42/bias/mAdam/dense_43/kernel/mAdam/dense_43/bias/mAdam/dense_44/kernel/mAdam/dense_44/bias/mAdam/embedding_26/embeddings/mAdam/embedding_24/embeddings/mAdam/embedding_25/embeddings/mAdam/embedding_29/embeddings/mAdam/embedding_27/embeddings/mAdam/embedding_28/embeddings/mAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/vAdam/dense_39/kernel/vAdam/dense_39/bias/vAdam/dense_38/kernel/vAdam/dense_38/bias/vAdam/dense_40/kernel/vAdam/dense_40/bias/vAdam/dense_41/kernel/vAdam/dense_41/bias/vAdam/dense_42/kernel/vAdam/dense_42/bias/vAdam/dense_43/kernel/vAdam/dense_43/bias/vAdam/dense_44/kernel/vAdam/dense_44/bias/vAdam/embedding_26/embeddings/vAdam/embedding_24/embeddings/vAdam/embedding_25/embeddings/vAdam/embedding_29/embeddings/vAdam/embedding_27/embeddings/vAdam/embedding_28/embeddings/v*^
TinW
U2S*
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
GPU2 *0J 8? *.
f)R'
%__inference__traced_restore_300121440??
?
d
H__inference_flatten_8_layer_call_and_return_conditional_losses_300118585

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
K__inference_embedding_27_layer_call_and_return_conditional_losses_300120884

inputs
embedding_lookup_300120878
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300120878Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300120878*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300120878*,
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
?
,__inference_dense_36_layer_call_fn_300120604

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_36_layer_call_and_return_conditional_losses_3001191252
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
G__inference_dense_41_layer_call_and_return_conditional_losses_300120692

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
?
?
+__inference_model_9_layer_call_fn_300120571

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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001189712
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
?	
?
K__inference_embedding_27_layer_call_and_return_conditional_losses_300118861

inputs
embedding_lookup_300118855
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300118855Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300118855*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300118855*,
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
+__inference_model_8_layer_call_fn_300120474

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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187902
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
?-
?
F__inference_model_9_layer_call_and_return_conditional_losses_300119016

inputs+
'tf_math_greater_equal_13_greaterequal_y
embedding_29_300118998
embedding_27_300119001
embedding_28_300119006
identity??$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_9_layer_call_and_return_conditional_losses_3001188112
flatten_9/PartitionedCall?
+tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_13/clip_by_value/Minimum/y?
)tf.clip_by_value_13/clip_by_value/MinimumMinimum"flatten_9/PartitionedCall:output:04tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_13/clip_by_value/Minimum?
#tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_13/clip_by_value/y?
!tf.clip_by_value_13/clip_by_valueMaximum-tf.clip_by_value_13/clip_by_value/Minimum:z:0,tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_13/clip_by_value?
#tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_9/FloorDiv/y?
!tf.compat.v1.floor_div_9/FloorDivFloorDiv%tf.clip_by_value_13/clip_by_value:z:0,tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_9/FloorDiv?
%tf.math.greater_equal_13/GreaterEqualGreaterEqual"flatten_9/PartitionedCall:output:0'tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_13/GreaterEqual?
tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_9/FloorMod/y?
tf.math.floormod_9/FloorModFloorMod%tf.clip_by_value_13/clip_by_value:z:0&tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_9/FloorMod?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_13/clip_by_value:z:0embedding_29_300118998*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_29_layer_call_and_return_conditional_losses_3001188392&
$embedding_29/StatefulPartitionedCall?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_9/FloorDiv:z:0embedding_27_300119001*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_27_layer_call_and_return_conditional_losses_3001188612&
$embedding_27/StatefulPartitionedCall?
tf.cast_13/CastCast)tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_13/Cast?
tf.__operators__.add_26/AddV2AddV2-embedding_29/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_26/AddV2?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_9/FloorMod:z:0embedding_28_300119006*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_28_layer_call_and_return_conditional_losses_3001188852&
$embedding_28/StatefulPartitionedCall?
tf.__operators__.add_27/AddV2AddV2!tf.__operators__.add_26/AddV2:z:0-embedding_28/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_27/AddV2?
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_9/ExpandDims/dim?
tf.expand_dims_9/ExpandDims
ExpandDimstf.cast_13/Cast:y:0(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_9/ExpandDims?
tf.math.multiply_9/MulMul!tf.__operators__.add_27/AddV2:z:0$tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_9/Mul?
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_9/Sum/reduction_indices?
tf.math.reduce_sum_9/SumSumtf.math.multiply_9/Mul:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_9/Sum?
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?Y
?

M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119392

cards0

cards1
bets+
'tf_math_greater_equal_14_greaterequal_y
model_8_300119061
model_8_300119063
model_8_300119065
model_8_300119067
model_9_300119096
model_9_300119098
model_9_300119100
model_9_300119102/
+tf_clip_by_value_14_clip_by_value_minimum_y'
#tf_clip_by_value_14_clip_by_value_y
dense_36_300119136
dense_36_300119138
dense_39_300119162
dense_39_300119164
dense_37_300119189
dense_37_300119191
dense_38_300119216
dense_38_300119218
dense_40_300119242
dense_40_300119244
dense_41_300119270
dense_41_300119272
dense_42_300119297
dense_42_300119299
dense_43_300119325
dense_43_300119327
normalize_4_300119360
normalize_4_300119362
dense_44_300119386
dense_44_300119388
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?model_8/StatefulPartitionedCall?model_9/StatefulPartitionedCall?#normalize_4/StatefulPartitionedCall?
%tf.math.greater_equal_14/GreaterEqualGreaterEqualbets'tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_14/GreaterEqual?
model_8/StatefulPartitionedCallStatefulPartitionedCallcards0model_8_300119061model_8_300119063model_8_300119065model_8_300119067*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187452!
model_8/StatefulPartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCallcards1model_9_300119096model_9_300119098model_9_300119100model_9_300119102*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001189712!
model_9/StatefulPartitionedCall?
)tf.clip_by_value_14/clip_by_value/MinimumMinimumbets+tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_14/clip_by_value/Minimum?
!tf.clip_by_value_14/clip_by_valueMaximum-tf.clip_by_value_14/clip_by_value/Minimum:z:0#tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_14/clip_by_value?
tf.cast_14/CastCast)tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_14/Castv
tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_12/concat/axis?
tf.concat_12/concatConcatV2(model_8/StatefulPartitionedCall:output:0(model_9/StatefulPartitionedCall:output:0!tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_12/concat
tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_13/concat/axis?
tf.concat_13/concatConcatV2%tf.clip_by_value_14/clip_by_value:z:0tf.cast_14/Cast:y:0!tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_13/concat?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalltf.concat_12/concat:output:0dense_36_300119136dense_36_300119138*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_36_layer_call_and_return_conditional_losses_3001191252"
 dense_36/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCalltf.concat_13/concat:output:0dense_39_300119162dense_39_300119164*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_39_layer_call_and_return_conditional_losses_3001191512"
 dense_39/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_300119189dense_37_300119191*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_37_layer_call_and_return_conditional_losses_3001191782"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_300119216dense_38_300119218*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_38_layer_call_and_return_conditional_losses_3001192052"
 dense_38/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_300119242dense_40_300119244*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_40_layer_call_and_return_conditional_losses_3001192312"
 dense_40/StatefulPartitionedCall
tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_14/concat/axis?
tf.concat_14/concatConcatV2)dense_38/StatefulPartitionedCall:output:0)dense_40/StatefulPartitionedCall:output:0!tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_14/concat?
 dense_41/StatefulPartitionedCallStatefulPartitionedCalltf.concat_14/concat:output:0dense_41_300119270dense_41_300119272*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_41_layer_call_and_return_conditional_losses_3001192592"
 dense_41/StatefulPartitionedCall?
tf.nn.relu_12/ReluRelu)dense_41/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_12/Relu?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_12/Relu:activations:0dense_42_300119297dense_42_300119299*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_42_layer_call_and_return_conditional_losses_3001192862"
 dense_42/StatefulPartitionedCall?
tf.__operators__.add_28/AddV2AddV2)dense_42/StatefulPartitionedCall:output:0 tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_28/AddV2?
tf.nn.relu_13/ReluRelu!tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_13/Relu?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_13/Relu:activations:0dense_43_300119325dense_43_300119327*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_43_layer_call_and_return_conditional_losses_3001193142"
 dense_43/StatefulPartitionedCall?
tf.__operators__.add_29/AddV2AddV2)dense_43/StatefulPartitionedCall:output:0 tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_29/AddV2?
tf.nn.relu_14/ReluRelu!tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_14/Relu?
#normalize_4/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_14/Relu:activations:0normalize_4_300119360normalize_4_300119362*
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
GPU2 *0J 8? *S
fNRL
J__inference_normalize_4_layer_call_and_return_conditional_losses_3001193492%
#normalize_4/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall,normalize_4/StatefulPartitionedCall:output:0dense_44_300119386dense_44_300119388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_44_layer_call_and_return_conditional_losses_3001193752"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall ^model_8/StatefulPartitionedCall ^model_9/StatefulPartitionedCall$^normalize_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall2J
#normalize_4/StatefulPartitionedCall#normalize_4/StatefulPartitionedCall:O K
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
?
v
0__inference_embedding_25_layer_call_fn_300120846

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
GPU2 *0J 8? *T
fORM
K__inference_embedding_25_layer_call_and_return_conditional_losses_3001186592
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
?	
?
K__inference_embedding_29_layer_call_and_return_conditional_losses_300120867

inputs
embedding_lookup_300120861
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300120861Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300120861*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300120861*,
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
2__inference_custom_model_4_layer_call_fn_300120364

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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_4_layer_call_and_return_conditional_losses_3001197422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
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
?
d
H__inference_flatten_8_layer_call_and_return_conditional_losses_300120790

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
?
v
0__inference_embedding_28_layer_call_fn_300120908

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
GPU2 *0J 8? *T
fORM
K__inference_embedding_28_layer_call_and_return_conditional_losses_3001188852
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
??
?
$__inference__wrapped_model_300118575

cards0

cards1
bets:
6custom_model_4_tf_math_greater_equal_14_greaterequal_yB
>custom_model_4_model_8_tf_math_greater_equal_12_greaterequal_yB
>custom_model_4_model_8_embedding_26_embedding_lookup_300118425B
>custom_model_4_model_8_embedding_24_embedding_lookup_300118431B
>custom_model_4_model_8_embedding_25_embedding_lookup_300118439B
>custom_model_4_model_9_tf_math_greater_equal_13_greaterequal_yB
>custom_model_4_model_9_embedding_29_embedding_lookup_300118463B
>custom_model_4_model_9_embedding_27_embedding_lookup_300118469B
>custom_model_4_model_9_embedding_28_embedding_lookup_300118477>
:custom_model_4_tf_clip_by_value_14_clip_by_value_minimum_y6
2custom_model_4_tf_clip_by_value_14_clip_by_value_y:
6custom_model_4_dense_36_matmul_readvariableop_resource;
7custom_model_4_dense_36_biasadd_readvariableop_resource:
6custom_model_4_dense_39_matmul_readvariableop_resource;
7custom_model_4_dense_39_biasadd_readvariableop_resource:
6custom_model_4_dense_37_matmul_readvariableop_resource;
7custom_model_4_dense_37_biasadd_readvariableop_resource:
6custom_model_4_dense_38_matmul_readvariableop_resource;
7custom_model_4_dense_38_biasadd_readvariableop_resource:
6custom_model_4_dense_40_matmul_readvariableop_resource;
7custom_model_4_dense_40_biasadd_readvariableop_resource:
6custom_model_4_dense_41_matmul_readvariableop_resource;
7custom_model_4_dense_41_biasadd_readvariableop_resource:
6custom_model_4_dense_42_matmul_readvariableop_resource;
7custom_model_4_dense_42_biasadd_readvariableop_resource:
6custom_model_4_dense_43_matmul_readvariableop_resource;
7custom_model_4_dense_43_biasadd_readvariableop_resourceN
Jcustom_model_4_normalize_4_normalization_4_reshape_readvariableop_resourceP
Lcustom_model_4_normalize_4_normalization_4_reshape_1_readvariableop_resource:
6custom_model_4_dense_44_matmul_readvariableop_resource;
7custom_model_4_dense_44_biasadd_readvariableop_resource
identity??.custom_model_4/dense_36/BiasAdd/ReadVariableOp?-custom_model_4/dense_36/MatMul/ReadVariableOp?.custom_model_4/dense_37/BiasAdd/ReadVariableOp?-custom_model_4/dense_37/MatMul/ReadVariableOp?.custom_model_4/dense_38/BiasAdd/ReadVariableOp?-custom_model_4/dense_38/MatMul/ReadVariableOp?.custom_model_4/dense_39/BiasAdd/ReadVariableOp?-custom_model_4/dense_39/MatMul/ReadVariableOp?.custom_model_4/dense_40/BiasAdd/ReadVariableOp?-custom_model_4/dense_40/MatMul/ReadVariableOp?.custom_model_4/dense_41/BiasAdd/ReadVariableOp?-custom_model_4/dense_41/MatMul/ReadVariableOp?.custom_model_4/dense_42/BiasAdd/ReadVariableOp?-custom_model_4/dense_42/MatMul/ReadVariableOp?.custom_model_4/dense_43/BiasAdd/ReadVariableOp?-custom_model_4/dense_43/MatMul/ReadVariableOp?.custom_model_4/dense_44/BiasAdd/ReadVariableOp?-custom_model_4/dense_44/MatMul/ReadVariableOp?4custom_model_4/model_8/embedding_24/embedding_lookup?4custom_model_4/model_8/embedding_25/embedding_lookup?4custom_model_4/model_8/embedding_26/embedding_lookup?4custom_model_4/model_9/embedding_27/embedding_lookup?4custom_model_4/model_9/embedding_28/embedding_lookup?4custom_model_4/model_9/embedding_29/embedding_lookup?Acustom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOp?Ccustom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOp?
4custom_model_4/tf.math.greater_equal_14/GreaterEqualGreaterEqualbets6custom_model_4_tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????26
4custom_model_4/tf.math.greater_equal_14/GreaterEqual?
&custom_model_4/model_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_4/model_8/flatten_8/Const?
(custom_model_4/model_8/flatten_8/ReshapeReshapecards0/custom_model_4/model_8/flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_4/model_8/flatten_8/Reshape?
Bcustom_model_4/model_8/tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2D
Bcustom_model_4/model_8/tf.clip_by_value_12/clip_by_value/Minimum/y?
@custom_model_4/model_8/tf.clip_by_value_12/clip_by_value/MinimumMinimum1custom_model_4/model_8/flatten_8/Reshape:output:0Kcustom_model_4/model_8/tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2B
@custom_model_4/model_8/tf.clip_by_value_12/clip_by_value/Minimum?
:custom_model_4/model_8/tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:custom_model_4/model_8/tf.clip_by_value_12/clip_by_value/y?
8custom_model_4/model_8/tf.clip_by_value_12/clip_by_valueMaximumDcustom_model_4/model_8/tf.clip_by_value_12/clip_by_value/Minimum:z:0Ccustom_model_4/model_8/tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_4/model_8/tf.clip_by_value_12/clip_by_value?
:custom_model_4/model_8/tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_4/model_8/tf.compat.v1.floor_div_8/FloorDiv/y?
8custom_model_4/model_8/tf.compat.v1.floor_div_8/FloorDivFloorDiv<custom_model_4/model_8/tf.clip_by_value_12/clip_by_value:z:0Ccustom_model_4/model_8/tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_4/model_8/tf.compat.v1.floor_div_8/FloorDiv?
<custom_model_4/model_8/tf.math.greater_equal_12/GreaterEqualGreaterEqual1custom_model_4/model_8/flatten_8/Reshape:output:0>custom_model_4_model_8_tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2>
<custom_model_4/model_8/tf.math.greater_equal_12/GreaterEqual?
4custom_model_4/model_8/tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_4/model_8/tf.math.floormod_8/FloorMod/y?
2custom_model_4/model_8/tf.math.floormod_8/FloorModFloorMod<custom_model_4/model_8/tf.clip_by_value_12/clip_by_value:z:0=custom_model_4/model_8/tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_4/model_8/tf.math.floormod_8/FloorMod?
(custom_model_4/model_8/embedding_26/CastCast<custom_model_4/model_8/tf.clip_by_value_12/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_4/model_8/embedding_26/Cast?
4custom_model_4/model_8/embedding_26/embedding_lookupResourceGather>custom_model_4_model_8_embedding_26_embedding_lookup_300118425,custom_model_4/model_8/embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_4/model_8/embedding_26/embedding_lookup/300118425*,
_output_shapes
:??????????*
dtype026
4custom_model_4/model_8/embedding_26/embedding_lookup?
=custom_model_4/model_8/embedding_26/embedding_lookup/IdentityIdentity=custom_model_4/model_8/embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_4/model_8/embedding_26/embedding_lookup/300118425*,
_output_shapes
:??????????2?
=custom_model_4/model_8/embedding_26/embedding_lookup/Identity?
?custom_model_4/model_8/embedding_26/embedding_lookup/Identity_1IdentityFcustom_model_4/model_8/embedding_26/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_4/model_8/embedding_26/embedding_lookup/Identity_1?
(custom_model_4/model_8/embedding_24/CastCast<custom_model_4/model_8/tf.compat.v1.floor_div_8/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_4/model_8/embedding_24/Cast?
4custom_model_4/model_8/embedding_24/embedding_lookupResourceGather>custom_model_4_model_8_embedding_24_embedding_lookup_300118431,custom_model_4/model_8/embedding_24/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_4/model_8/embedding_24/embedding_lookup/300118431*,
_output_shapes
:??????????*
dtype026
4custom_model_4/model_8/embedding_24/embedding_lookup?
=custom_model_4/model_8/embedding_24/embedding_lookup/IdentityIdentity=custom_model_4/model_8/embedding_24/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_4/model_8/embedding_24/embedding_lookup/300118431*,
_output_shapes
:??????????2?
=custom_model_4/model_8/embedding_24/embedding_lookup/Identity?
?custom_model_4/model_8/embedding_24/embedding_lookup/Identity_1IdentityFcustom_model_4/model_8/embedding_24/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_4/model_8/embedding_24/embedding_lookup/Identity_1?
&custom_model_4/model_8/tf.cast_12/CastCast@custom_model_4/model_8/tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2(
&custom_model_4/model_8/tf.cast_12/Cast?
4custom_model_4/model_8/tf.__operators__.add_24/AddV2AddV2Hcustom_model_4/model_8/embedding_26/embedding_lookup/Identity_1:output:0Hcustom_model_4/model_8/embedding_24/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_4/model_8/tf.__operators__.add_24/AddV2?
(custom_model_4/model_8/embedding_25/CastCast6custom_model_4/model_8/tf.math.floormod_8/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_4/model_8/embedding_25/Cast?
4custom_model_4/model_8/embedding_25/embedding_lookupResourceGather>custom_model_4_model_8_embedding_25_embedding_lookup_300118439,custom_model_4/model_8/embedding_25/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_4/model_8/embedding_25/embedding_lookup/300118439*,
_output_shapes
:??????????*
dtype026
4custom_model_4/model_8/embedding_25/embedding_lookup?
=custom_model_4/model_8/embedding_25/embedding_lookup/IdentityIdentity=custom_model_4/model_8/embedding_25/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_4/model_8/embedding_25/embedding_lookup/300118439*,
_output_shapes
:??????????2?
=custom_model_4/model_8/embedding_25/embedding_lookup/Identity?
?custom_model_4/model_8/embedding_25/embedding_lookup/Identity_1IdentityFcustom_model_4/model_8/embedding_25/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_4/model_8/embedding_25/embedding_lookup/Identity_1?
4custom_model_4/model_8/tf.__operators__.add_25/AddV2AddV28custom_model_4/model_8/tf.__operators__.add_24/AddV2:z:0Hcustom_model_4/model_8/embedding_25/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_4/model_8/tf.__operators__.add_25/AddV2?
6custom_model_4/model_8/tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_4/model_8/tf.expand_dims_8/ExpandDims/dim?
2custom_model_4/model_8/tf.expand_dims_8/ExpandDims
ExpandDims*custom_model_4/model_8/tf.cast_12/Cast:y:0?custom_model_4/model_8/tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_4/model_8/tf.expand_dims_8/ExpandDims?
-custom_model_4/model_8/tf.math.multiply_8/MulMul8custom_model_4/model_8/tf.__operators__.add_25/AddV2:z:0;custom_model_4/model_8/tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_4/model_8/tf.math.multiply_8/Mul?
Acustom_model_4/model_8/tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_4/model_8/tf.math.reduce_sum_8/Sum/reduction_indices?
/custom_model_4/model_8/tf.math.reduce_sum_8/SumSum1custom_model_4/model_8/tf.math.multiply_8/Mul:z:0Jcustom_model_4/model_8/tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_4/model_8/tf.math.reduce_sum_8/Sum?
&custom_model_4/model_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_4/model_9/flatten_9/Const?
(custom_model_4/model_9/flatten_9/ReshapeReshapecards1/custom_model_4/model_9/flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_4/model_9/flatten_9/Reshape?
Bcustom_model_4/model_9/tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2D
Bcustom_model_4/model_9/tf.clip_by_value_13/clip_by_value/Minimum/y?
@custom_model_4/model_9/tf.clip_by_value_13/clip_by_value/MinimumMinimum1custom_model_4/model_9/flatten_9/Reshape:output:0Kcustom_model_4/model_9/tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2B
@custom_model_4/model_9/tf.clip_by_value_13/clip_by_value/Minimum?
:custom_model_4/model_9/tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:custom_model_4/model_9/tf.clip_by_value_13/clip_by_value/y?
8custom_model_4/model_9/tf.clip_by_value_13/clip_by_valueMaximumDcustom_model_4/model_9/tf.clip_by_value_13/clip_by_value/Minimum:z:0Ccustom_model_4/model_9/tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_4/model_9/tf.clip_by_value_13/clip_by_value?
:custom_model_4/model_9/tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_4/model_9/tf.compat.v1.floor_div_9/FloorDiv/y?
8custom_model_4/model_9/tf.compat.v1.floor_div_9/FloorDivFloorDiv<custom_model_4/model_9/tf.clip_by_value_13/clip_by_value:z:0Ccustom_model_4/model_9/tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_4/model_9/tf.compat.v1.floor_div_9/FloorDiv?
<custom_model_4/model_9/tf.math.greater_equal_13/GreaterEqualGreaterEqual1custom_model_4/model_9/flatten_9/Reshape:output:0>custom_model_4_model_9_tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2>
<custom_model_4/model_9/tf.math.greater_equal_13/GreaterEqual?
4custom_model_4/model_9/tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_4/model_9/tf.math.floormod_9/FloorMod/y?
2custom_model_4/model_9/tf.math.floormod_9/FloorModFloorMod<custom_model_4/model_9/tf.clip_by_value_13/clip_by_value:z:0=custom_model_4/model_9/tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_4/model_9/tf.math.floormod_9/FloorMod?
(custom_model_4/model_9/embedding_29/CastCast<custom_model_4/model_9/tf.clip_by_value_13/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_4/model_9/embedding_29/Cast?
4custom_model_4/model_9/embedding_29/embedding_lookupResourceGather>custom_model_4_model_9_embedding_29_embedding_lookup_300118463,custom_model_4/model_9/embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_4/model_9/embedding_29/embedding_lookup/300118463*,
_output_shapes
:??????????*
dtype026
4custom_model_4/model_9/embedding_29/embedding_lookup?
=custom_model_4/model_9/embedding_29/embedding_lookup/IdentityIdentity=custom_model_4/model_9/embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_4/model_9/embedding_29/embedding_lookup/300118463*,
_output_shapes
:??????????2?
=custom_model_4/model_9/embedding_29/embedding_lookup/Identity?
?custom_model_4/model_9/embedding_29/embedding_lookup/Identity_1IdentityFcustom_model_4/model_9/embedding_29/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_4/model_9/embedding_29/embedding_lookup/Identity_1?
(custom_model_4/model_9/embedding_27/CastCast<custom_model_4/model_9/tf.compat.v1.floor_div_9/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_4/model_9/embedding_27/Cast?
4custom_model_4/model_9/embedding_27/embedding_lookupResourceGather>custom_model_4_model_9_embedding_27_embedding_lookup_300118469,custom_model_4/model_9/embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_4/model_9/embedding_27/embedding_lookup/300118469*,
_output_shapes
:??????????*
dtype026
4custom_model_4/model_9/embedding_27/embedding_lookup?
=custom_model_4/model_9/embedding_27/embedding_lookup/IdentityIdentity=custom_model_4/model_9/embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_4/model_9/embedding_27/embedding_lookup/300118469*,
_output_shapes
:??????????2?
=custom_model_4/model_9/embedding_27/embedding_lookup/Identity?
?custom_model_4/model_9/embedding_27/embedding_lookup/Identity_1IdentityFcustom_model_4/model_9/embedding_27/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_4/model_9/embedding_27/embedding_lookup/Identity_1?
&custom_model_4/model_9/tf.cast_13/CastCast@custom_model_4/model_9/tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2(
&custom_model_4/model_9/tf.cast_13/Cast?
4custom_model_4/model_9/tf.__operators__.add_26/AddV2AddV2Hcustom_model_4/model_9/embedding_29/embedding_lookup/Identity_1:output:0Hcustom_model_4/model_9/embedding_27/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_4/model_9/tf.__operators__.add_26/AddV2?
(custom_model_4/model_9/embedding_28/CastCast6custom_model_4/model_9/tf.math.floormod_9/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_4/model_9/embedding_28/Cast?
4custom_model_4/model_9/embedding_28/embedding_lookupResourceGather>custom_model_4_model_9_embedding_28_embedding_lookup_300118477,custom_model_4/model_9/embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_4/model_9/embedding_28/embedding_lookup/300118477*,
_output_shapes
:??????????*
dtype026
4custom_model_4/model_9/embedding_28/embedding_lookup?
=custom_model_4/model_9/embedding_28/embedding_lookup/IdentityIdentity=custom_model_4/model_9/embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_4/model_9/embedding_28/embedding_lookup/300118477*,
_output_shapes
:??????????2?
=custom_model_4/model_9/embedding_28/embedding_lookup/Identity?
?custom_model_4/model_9/embedding_28/embedding_lookup/Identity_1IdentityFcustom_model_4/model_9/embedding_28/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_4/model_9/embedding_28/embedding_lookup/Identity_1?
4custom_model_4/model_9/tf.__operators__.add_27/AddV2AddV28custom_model_4/model_9/tf.__operators__.add_26/AddV2:z:0Hcustom_model_4/model_9/embedding_28/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_4/model_9/tf.__operators__.add_27/AddV2?
6custom_model_4/model_9/tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_4/model_9/tf.expand_dims_9/ExpandDims/dim?
2custom_model_4/model_9/tf.expand_dims_9/ExpandDims
ExpandDims*custom_model_4/model_9/tf.cast_13/Cast:y:0?custom_model_4/model_9/tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_4/model_9/tf.expand_dims_9/ExpandDims?
-custom_model_4/model_9/tf.math.multiply_9/MulMul8custom_model_4/model_9/tf.__operators__.add_27/AddV2:z:0;custom_model_4/model_9/tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_4/model_9/tf.math.multiply_9/Mul?
Acustom_model_4/model_9/tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_4/model_9/tf.math.reduce_sum_9/Sum/reduction_indices?
/custom_model_4/model_9/tf.math.reduce_sum_9/SumSum1custom_model_4/model_9/tf.math.multiply_9/Mul:z:0Jcustom_model_4/model_9/tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_4/model_9/tf.math.reduce_sum_9/Sum?
8custom_model_4/tf.clip_by_value_14/clip_by_value/MinimumMinimumbets:custom_model_4_tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2:
8custom_model_4/tf.clip_by_value_14/clip_by_value/Minimum?
0custom_model_4/tf.clip_by_value_14/clip_by_valueMaximum<custom_model_4/tf.clip_by_value_14/clip_by_value/Minimum:z:02custom_model_4_tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????22
0custom_model_4/tf.clip_by_value_14/clip_by_value?
custom_model_4/tf.cast_14/CastCast8custom_model_4/tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2 
custom_model_4/tf.cast_14/Cast?
'custom_model_4/tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'custom_model_4/tf.concat_12/concat/axis?
"custom_model_4/tf.concat_12/concatConcatV28custom_model_4/model_8/tf.math.reduce_sum_8/Sum:output:08custom_model_4/model_9/tf.math.reduce_sum_9/Sum:output:00custom_model_4/tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"custom_model_4/tf.concat_12/concat?
'custom_model_4/tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_4/tf.concat_13/concat/axis?
"custom_model_4/tf.concat_13/concatConcatV24custom_model_4/tf.clip_by_value_14/clip_by_value:z:0"custom_model_4/tf.cast_14/Cast:y:00custom_model_4/tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2$
"custom_model_4/tf.concat_13/concat?
-custom_model_4/dense_36/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_36/MatMul/ReadVariableOp?
custom_model_4/dense_36/MatMulMatMul+custom_model_4/tf.concat_12/concat:output:05custom_model_4/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_36/MatMul?
.custom_model_4/dense_36/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_36/BiasAdd/ReadVariableOp?
custom_model_4/dense_36/BiasAddBiasAdd(custom_model_4/dense_36/MatMul:product:06custom_model_4/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_36/BiasAdd?
custom_model_4/dense_36/ReluRelu(custom_model_4/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_4/dense_36/Relu?
-custom_model_4/dense_39/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_39_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_4/dense_39/MatMul/ReadVariableOp?
custom_model_4/dense_39/MatMulMatMul+custom_model_4/tf.concat_13/concat:output:05custom_model_4/dense_39/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_39/MatMul?
.custom_model_4/dense_39/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_39/BiasAdd/ReadVariableOp?
custom_model_4/dense_39/BiasAddBiasAdd(custom_model_4/dense_39/MatMul:product:06custom_model_4/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_39/BiasAdd?
-custom_model_4/dense_37/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_37/MatMul/ReadVariableOp?
custom_model_4/dense_37/MatMulMatMul*custom_model_4/dense_36/Relu:activations:05custom_model_4/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_37/MatMul?
.custom_model_4/dense_37/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_37/BiasAdd/ReadVariableOp?
custom_model_4/dense_37/BiasAddBiasAdd(custom_model_4/dense_37/MatMul:product:06custom_model_4/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_37/BiasAdd?
custom_model_4/dense_37/ReluRelu(custom_model_4/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_4/dense_37/Relu?
-custom_model_4/dense_38/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_38_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_38/MatMul/ReadVariableOp?
custom_model_4/dense_38/MatMulMatMul*custom_model_4/dense_37/Relu:activations:05custom_model_4/dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_38/MatMul?
.custom_model_4/dense_38/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_38/BiasAdd/ReadVariableOp?
custom_model_4/dense_38/BiasAddBiasAdd(custom_model_4/dense_38/MatMul:product:06custom_model_4/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_38/BiasAdd?
custom_model_4/dense_38/ReluRelu(custom_model_4/dense_38/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_4/dense_38/Relu?
-custom_model_4/dense_40/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_40_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_40/MatMul/ReadVariableOp?
custom_model_4/dense_40/MatMulMatMul(custom_model_4/dense_39/BiasAdd:output:05custom_model_4/dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_40/MatMul?
.custom_model_4/dense_40/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_40/BiasAdd/ReadVariableOp?
custom_model_4/dense_40/BiasAddBiasAdd(custom_model_4/dense_40/MatMul:product:06custom_model_4/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_40/BiasAdd?
'custom_model_4/tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_4/tf.concat_14/concat/axis?
"custom_model_4/tf.concat_14/concatConcatV2*custom_model_4/dense_38/Relu:activations:0(custom_model_4/dense_40/BiasAdd:output:00custom_model_4/tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"custom_model_4/tf.concat_14/concat?
-custom_model_4/dense_41/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_41_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_41/MatMul/ReadVariableOp?
custom_model_4/dense_41/MatMulMatMul+custom_model_4/tf.concat_14/concat:output:05custom_model_4/dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_41/MatMul?
.custom_model_4/dense_41/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_41/BiasAdd/ReadVariableOp?
custom_model_4/dense_41/BiasAddBiasAdd(custom_model_4/dense_41/MatMul:product:06custom_model_4/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_41/BiasAdd?
!custom_model_4/tf.nn.relu_12/ReluRelu(custom_model_4/dense_41/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2#
!custom_model_4/tf.nn.relu_12/Relu?
-custom_model_4/dense_42/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_42_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_42/MatMul/ReadVariableOp?
custom_model_4/dense_42/MatMulMatMul/custom_model_4/tf.nn.relu_12/Relu:activations:05custom_model_4/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_42/MatMul?
.custom_model_4/dense_42/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_42/BiasAdd/ReadVariableOp?
custom_model_4/dense_42/BiasAddBiasAdd(custom_model_4/dense_42/MatMul:product:06custom_model_4/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_42/BiasAdd?
,custom_model_4/tf.__operators__.add_28/AddV2AddV2(custom_model_4/dense_42/BiasAdd:output:0/custom_model_4/tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_4/tf.__operators__.add_28/AddV2?
!custom_model_4/tf.nn.relu_13/ReluRelu0custom_model_4/tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_4/tf.nn.relu_13/Relu?
-custom_model_4/dense_43/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_4/dense_43/MatMul/ReadVariableOp?
custom_model_4/dense_43/MatMulMatMul/custom_model_4/tf.nn.relu_13/Relu:activations:05custom_model_4/dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_4/dense_43/MatMul?
.custom_model_4/dense_43/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_4/dense_43/BiasAdd/ReadVariableOp?
custom_model_4/dense_43/BiasAddBiasAdd(custom_model_4/dense_43/MatMul:product:06custom_model_4/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_4/dense_43/BiasAdd?
,custom_model_4/tf.__operators__.add_29/AddV2AddV2(custom_model_4/dense_43/BiasAdd:output:0/custom_model_4/tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_4/tf.__operators__.add_29/AddV2?
!custom_model_4/tf.nn.relu_14/ReluRelu0custom_model_4/tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_4/tf.nn.relu_14/Relu?
Acustom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOpReadVariableOpJcustom_model_4_normalize_4_normalization_4_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acustom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOp?
8custom_model_4/normalize_4/normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_4/normalize_4/normalization_4/Reshape/shape?
2custom_model_4/normalize_4/normalization_4/ReshapeReshapeIcustom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOp:value:0Acustom_model_4/normalize_4/normalization_4/Reshape/shape:output:0*
T0*
_output_shapes
:	?24
2custom_model_4/normalize_4/normalization_4/Reshape?
Ccustom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_4_normalize_4_normalization_4_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Ccustom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOp?
:custom_model_4/normalize_4/normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_4/normalize_4/normalization_4/Reshape_1/shape?
4custom_model_4/normalize_4/normalization_4/Reshape_1ReshapeKcustom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOp:value:0Ccustom_model_4/normalize_4/normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?26
4custom_model_4/normalize_4/normalization_4/Reshape_1?
.custom_model_4/normalize_4/normalization_4/subSub/custom_model_4/tf.nn.relu_14/Relu:activations:0;custom_model_4/normalize_4/normalization_4/Reshape:output:0*
T0*(
_output_shapes
:??????????20
.custom_model_4/normalize_4/normalization_4/sub?
/custom_model_4/normalize_4/normalization_4/SqrtSqrt=custom_model_4/normalize_4/normalization_4/Reshape_1:output:0*
T0*
_output_shapes
:	?21
/custom_model_4/normalize_4/normalization_4/Sqrt?
4custom_model_4/normalize_4/normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???326
4custom_model_4/normalize_4/normalization_4/Maximum/y?
2custom_model_4/normalize_4/normalization_4/MaximumMaximum3custom_model_4/normalize_4/normalization_4/Sqrt:y:0=custom_model_4/normalize_4/normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:	?24
2custom_model_4/normalize_4/normalization_4/Maximum?
2custom_model_4/normalize_4/normalization_4/truedivRealDiv2custom_model_4/normalize_4/normalization_4/sub:z:06custom_model_4/normalize_4/normalization_4/Maximum:z:0*
T0*(
_output_shapes
:??????????24
2custom_model_4/normalize_4/normalization_4/truediv?
-custom_model_4/dense_44/MatMul/ReadVariableOpReadVariableOp6custom_model_4_dense_44_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_4/dense_44/MatMul/ReadVariableOp?
custom_model_4/dense_44/MatMulMatMul6custom_model_4/normalize_4/normalization_4/truediv:z:05custom_model_4/dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
custom_model_4/dense_44/MatMul?
.custom_model_4/dense_44/BiasAdd/ReadVariableOpReadVariableOp7custom_model_4_dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_4/dense_44/BiasAdd/ReadVariableOp?
custom_model_4/dense_44/BiasAddBiasAdd(custom_model_4/dense_44/MatMul:product:06custom_model_4/dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
custom_model_4/dense_44/BiasAdd?
IdentityIdentity(custom_model_4/dense_44/BiasAdd:output:0/^custom_model_4/dense_36/BiasAdd/ReadVariableOp.^custom_model_4/dense_36/MatMul/ReadVariableOp/^custom_model_4/dense_37/BiasAdd/ReadVariableOp.^custom_model_4/dense_37/MatMul/ReadVariableOp/^custom_model_4/dense_38/BiasAdd/ReadVariableOp.^custom_model_4/dense_38/MatMul/ReadVariableOp/^custom_model_4/dense_39/BiasAdd/ReadVariableOp.^custom_model_4/dense_39/MatMul/ReadVariableOp/^custom_model_4/dense_40/BiasAdd/ReadVariableOp.^custom_model_4/dense_40/MatMul/ReadVariableOp/^custom_model_4/dense_41/BiasAdd/ReadVariableOp.^custom_model_4/dense_41/MatMul/ReadVariableOp/^custom_model_4/dense_42/BiasAdd/ReadVariableOp.^custom_model_4/dense_42/MatMul/ReadVariableOp/^custom_model_4/dense_43/BiasAdd/ReadVariableOp.^custom_model_4/dense_43/MatMul/ReadVariableOp/^custom_model_4/dense_44/BiasAdd/ReadVariableOp.^custom_model_4/dense_44/MatMul/ReadVariableOp5^custom_model_4/model_8/embedding_24/embedding_lookup5^custom_model_4/model_8/embedding_25/embedding_lookup5^custom_model_4/model_8/embedding_26/embedding_lookup5^custom_model_4/model_9/embedding_27/embedding_lookup5^custom_model_4/model_9/embedding_28/embedding_lookup5^custom_model_4/model_9/embedding_29/embedding_lookupB^custom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOpD^custom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2`
.custom_model_4/dense_36/BiasAdd/ReadVariableOp.custom_model_4/dense_36/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_36/MatMul/ReadVariableOp-custom_model_4/dense_36/MatMul/ReadVariableOp2`
.custom_model_4/dense_37/BiasAdd/ReadVariableOp.custom_model_4/dense_37/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_37/MatMul/ReadVariableOp-custom_model_4/dense_37/MatMul/ReadVariableOp2`
.custom_model_4/dense_38/BiasAdd/ReadVariableOp.custom_model_4/dense_38/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_38/MatMul/ReadVariableOp-custom_model_4/dense_38/MatMul/ReadVariableOp2`
.custom_model_4/dense_39/BiasAdd/ReadVariableOp.custom_model_4/dense_39/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_39/MatMul/ReadVariableOp-custom_model_4/dense_39/MatMul/ReadVariableOp2`
.custom_model_4/dense_40/BiasAdd/ReadVariableOp.custom_model_4/dense_40/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_40/MatMul/ReadVariableOp-custom_model_4/dense_40/MatMul/ReadVariableOp2`
.custom_model_4/dense_41/BiasAdd/ReadVariableOp.custom_model_4/dense_41/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_41/MatMul/ReadVariableOp-custom_model_4/dense_41/MatMul/ReadVariableOp2`
.custom_model_4/dense_42/BiasAdd/ReadVariableOp.custom_model_4/dense_42/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_42/MatMul/ReadVariableOp-custom_model_4/dense_42/MatMul/ReadVariableOp2`
.custom_model_4/dense_43/BiasAdd/ReadVariableOp.custom_model_4/dense_43/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_43/MatMul/ReadVariableOp-custom_model_4/dense_43/MatMul/ReadVariableOp2`
.custom_model_4/dense_44/BiasAdd/ReadVariableOp.custom_model_4/dense_44/BiasAdd/ReadVariableOp2^
-custom_model_4/dense_44/MatMul/ReadVariableOp-custom_model_4/dense_44/MatMul/ReadVariableOp2l
4custom_model_4/model_8/embedding_24/embedding_lookup4custom_model_4/model_8/embedding_24/embedding_lookup2l
4custom_model_4/model_8/embedding_25/embedding_lookup4custom_model_4/model_8/embedding_25/embedding_lookup2l
4custom_model_4/model_8/embedding_26/embedding_lookup4custom_model_4/model_8/embedding_26/embedding_lookup2l
4custom_model_4/model_9/embedding_27/embedding_lookup4custom_model_4/model_9/embedding_27/embedding_lookup2l
4custom_model_4/model_9/embedding_28/embedding_lookup4custom_model_4/model_9/embedding_28/embedding_lookup2l
4custom_model_4/model_9/embedding_29/embedding_lookup4custom_model_4/model_9/embedding_29/embedding_lookup2?
Acustom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOpAcustom_model_4/normalize_4/normalization_4/Reshape/ReadVariableOp2?
Ccustom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOpCcustom_model_4/normalize_4/normalization_4/Reshape_1/ReadVariableOp:O K
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
?
v
0__inference_embedding_27_layer_call_fn_300120891

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
GPU2 *0J 8? *T
fORM
K__inference_embedding_27_layer_call_and_return_conditional_losses_3001188612
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
+__inference_model_9_layer_call_fn_300120584

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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001190162
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
?	
?
G__inference_dense_36_layer_call_and_return_conditional_losses_300120595

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
G__inference_dense_40_layer_call_and_return_conditional_losses_300119231

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
?9
?
F__inference_model_8_layer_call_and_return_conditional_losses_300120448

inputs+
'tf_math_greater_equal_12_greaterequal_y+
'embedding_26_embedding_lookup_300120422+
'embedding_24_embedding_lookup_300120428+
'embedding_25_embedding_lookup_300120436
identity??embedding_24/embedding_lookup?embedding_25/embedding_lookup?embedding_26/embedding_lookups
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshapeinputsflatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_8/Reshape?
+tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_12/clip_by_value/Minimum/y?
)tf.clip_by_value_12/clip_by_value/MinimumMinimumflatten_8/Reshape:output:04tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_12/clip_by_value/Minimum?
#tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_12/clip_by_value/y?
!tf.clip_by_value_12/clip_by_valueMaximum-tf.clip_by_value_12/clip_by_value/Minimum:z:0,tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_12/clip_by_value?
#tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_8/FloorDiv/y?
!tf.compat.v1.floor_div_8/FloorDivFloorDiv%tf.clip_by_value_12/clip_by_value:z:0,tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_8/FloorDiv?
%tf.math.greater_equal_12/GreaterEqualGreaterEqualflatten_8/Reshape:output:0'tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_12/GreaterEqual?
tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_8/FloorMod/y?
tf.math.floormod_8/FloorModFloorMod%tf.clip_by_value_12/clip_by_value:z:0&tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_8/FloorMod?
embedding_26/CastCast%tf.clip_by_value_12/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_26/Cast?
embedding_26/embedding_lookupResourceGather'embedding_26_embedding_lookup_300120422embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_26/embedding_lookup/300120422*,
_output_shapes
:??????????*
dtype02
embedding_26/embedding_lookup?
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_26/embedding_lookup/300120422*,
_output_shapes
:??????????2(
&embedding_26/embedding_lookup/Identity?
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_26/embedding_lookup/Identity_1?
embedding_24/CastCast%tf.compat.v1.floor_div_8/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_24/Cast?
embedding_24/embedding_lookupResourceGather'embedding_24_embedding_lookup_300120428embedding_24/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_24/embedding_lookup/300120428*,
_output_shapes
:??????????*
dtype02
embedding_24/embedding_lookup?
&embedding_24/embedding_lookup/IdentityIdentity&embedding_24/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_24/embedding_lookup/300120428*,
_output_shapes
:??????????2(
&embedding_24/embedding_lookup/Identity?
(embedding_24/embedding_lookup/Identity_1Identity/embedding_24/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_24/embedding_lookup/Identity_1?
tf.cast_12/CastCast)tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_12/Cast?
tf.__operators__.add_24/AddV2AddV21embedding_26/embedding_lookup/Identity_1:output:01embedding_24/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_24/AddV2?
embedding_25/CastCasttf.math.floormod_8/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_25/Cast?
embedding_25/embedding_lookupResourceGather'embedding_25_embedding_lookup_300120436embedding_25/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_25/embedding_lookup/300120436*,
_output_shapes
:??????????*
dtype02
embedding_25/embedding_lookup?
&embedding_25/embedding_lookup/IdentityIdentity&embedding_25/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_25/embedding_lookup/300120436*,
_output_shapes
:??????????2(
&embedding_25/embedding_lookup/Identity?
(embedding_25/embedding_lookup/Identity_1Identity/embedding_25/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_25/embedding_lookup/Identity_1?
tf.__operators__.add_25/AddV2AddV2!tf.__operators__.add_24/AddV2:z:01embedding_25/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_25/AddV2?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimstf.cast_12/Cast:y:0(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_8/ExpandDims?
tf.math.multiply_8/MulMul!tf.__operators__.add_25/AddV2:z:0$tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_8/Mul?
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_8/Sum/reduction_indices?
tf.math.reduce_sum_8/SumSumtf.math.multiply_8/Mul:z:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_8/Sum?
IdentityIdentity!tf.math.reduce_sum_8/Sum:output:0^embedding_24/embedding_lookup^embedding_25/embedding_lookup^embedding_26/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_24/embedding_lookupembedding_24/embedding_lookup2>
embedding_25/embedding_lookupembedding_25/embedding_lookup2>
embedding_26/embedding_lookupembedding_26/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
G__inference_dense_37_layer_call_and_return_conditional_losses_300119178

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
K__inference_embedding_26_layer_call_and_return_conditional_losses_300120805

inputs
embedding_lookup_300120799
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300120799Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300120799*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300120799*,
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
?
?
J__inference_normalize_4_layer_call_and_return_conditional_losses_300120756
x3
/normalization_4_reshape_readvariableop_resource5
1normalization_4_reshape_1_readvariableop_resource
identity??&normalization_4/Reshape/ReadVariableOp?(normalization_4/Reshape_1/ReadVariableOp?
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_4/Reshape/ReadVariableOp?
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape?
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_4/Reshape?
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_4/Reshape_1/ReadVariableOp?
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape?
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_4/Reshape_1?
normalization_4/subSubx normalization_4/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_4/sub?
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_4/truediv?
IdentityIdentitynormalization_4/truediv:z:0'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
G__inference_dense_39_layer_call_and_return_conditional_losses_300120634

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
F__inference_model_9_layer_call_and_return_conditional_losses_300118904
input_10+
'tf_math_greater_equal_13_greaterequal_y
embedding_29_300118848
embedding_27_300118870
embedding_28_300118894
identity??$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallinput_10*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_9_layer_call_and_return_conditional_losses_3001188112
flatten_9/PartitionedCall?
+tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_13/clip_by_value/Minimum/y?
)tf.clip_by_value_13/clip_by_value/MinimumMinimum"flatten_9/PartitionedCall:output:04tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_13/clip_by_value/Minimum?
#tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_13/clip_by_value/y?
!tf.clip_by_value_13/clip_by_valueMaximum-tf.clip_by_value_13/clip_by_value/Minimum:z:0,tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_13/clip_by_value?
#tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_9/FloorDiv/y?
!tf.compat.v1.floor_div_9/FloorDivFloorDiv%tf.clip_by_value_13/clip_by_value:z:0,tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_9/FloorDiv?
%tf.math.greater_equal_13/GreaterEqualGreaterEqual"flatten_9/PartitionedCall:output:0'tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_13/GreaterEqual?
tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_9/FloorMod/y?
tf.math.floormod_9/FloorModFloorMod%tf.clip_by_value_13/clip_by_value:z:0&tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_9/FloorMod?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_13/clip_by_value:z:0embedding_29_300118848*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_29_layer_call_and_return_conditional_losses_3001188392&
$embedding_29/StatefulPartitionedCall?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_9/FloorDiv:z:0embedding_27_300118870*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_27_layer_call_and_return_conditional_losses_3001188612&
$embedding_27/StatefulPartitionedCall?
tf.cast_13/CastCast)tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_13/Cast?
tf.__operators__.add_26/AddV2AddV2-embedding_29/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_26/AddV2?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_9/FloorMod:z:0embedding_28_300118894*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_28_layer_call_and_return_conditional_losses_3001188852&
$embedding_28/StatefulPartitionedCall?
tf.__operators__.add_27/AddV2AddV2!tf.__operators__.add_26/AddV2:z:0-embedding_28/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_27/AddV2?
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_9/ExpandDims/dim?
tf.expand_dims_9/ExpandDims
ExpandDimstf.cast_13/Cast:y:0(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_9/ExpandDims?
tf.math.multiply_9/MulMul!tf.__operators__.add_27/AddV2:z:0$tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_9/Mul?
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_9/Sum/reduction_indices?
tf.math.reduce_sum_9/SumSumtf.math.multiply_9/Mul:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_9/Sum?
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:

_output_shapes
: 
?-
?
F__inference_model_8_layer_call_and_return_conditional_losses_300118790

inputs+
'tf_math_greater_equal_12_greaterequal_y
embedding_26_300118772
embedding_24_300118775
embedding_25_300118780
identity??$embedding_24/StatefulPartitionedCall?$embedding_25/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_3001185852
flatten_8/PartitionedCall?
+tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_12/clip_by_value/Minimum/y?
)tf.clip_by_value_12/clip_by_value/MinimumMinimum"flatten_8/PartitionedCall:output:04tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_12/clip_by_value/Minimum?
#tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_12/clip_by_value/y?
!tf.clip_by_value_12/clip_by_valueMaximum-tf.clip_by_value_12/clip_by_value/Minimum:z:0,tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_12/clip_by_value?
#tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_8/FloorDiv/y?
!tf.compat.v1.floor_div_8/FloorDivFloorDiv%tf.clip_by_value_12/clip_by_value:z:0,tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_8/FloorDiv?
%tf.math.greater_equal_12/GreaterEqualGreaterEqual"flatten_8/PartitionedCall:output:0'tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_12/GreaterEqual?
tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_8/FloorMod/y?
tf.math.floormod_8/FloorModFloorMod%tf.clip_by_value_12/clip_by_value:z:0&tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_8/FloorMod?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_12/clip_by_value:z:0embedding_26_300118772*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_26_layer_call_and_return_conditional_losses_3001186132&
$embedding_26/StatefulPartitionedCall?
$embedding_24/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_8/FloorDiv:z:0embedding_24_300118775*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_24_layer_call_and_return_conditional_losses_3001186352&
$embedding_24/StatefulPartitionedCall?
tf.cast_12/CastCast)tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_12/Cast?
tf.__operators__.add_24/AddV2AddV2-embedding_26/StatefulPartitionedCall:output:0-embedding_24/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_24/AddV2?
$embedding_25/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_8/FloorMod:z:0embedding_25_300118780*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_25_layer_call_and_return_conditional_losses_3001186592&
$embedding_25/StatefulPartitionedCall?
tf.__operators__.add_25/AddV2AddV2!tf.__operators__.add_24/AddV2:z:0-embedding_25/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_25/AddV2?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimstf.cast_12/Cast:y:0(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_8/ExpandDims?
tf.math.multiply_8/MulMul!tf.__operators__.add_25/AddV2:z:0$tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_8/Mul?
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_8/Sum/reduction_indices?
tf.math.reduce_sum_8/SumSumtf.math.multiply_8/Mul:z:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_8/Sum?
IdentityIdentity!tf.math.reduce_sum_8/Sum:output:0%^embedding_24/StatefulPartitionedCall%^embedding_25/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_24/StatefulPartitionedCall$embedding_24/StatefulPartitionedCall2L
$embedding_25/StatefulPartitionedCall$embedding_25/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?-
?
F__inference_model_9_layer_call_and_return_conditional_losses_300118971

inputs+
'tf_math_greater_equal_13_greaterequal_y
embedding_29_300118953
embedding_27_300118956
embedding_28_300118961
identity??$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_9_layer_call_and_return_conditional_losses_3001188112
flatten_9/PartitionedCall?
+tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_13/clip_by_value/Minimum/y?
)tf.clip_by_value_13/clip_by_value/MinimumMinimum"flatten_9/PartitionedCall:output:04tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_13/clip_by_value/Minimum?
#tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_13/clip_by_value/y?
!tf.clip_by_value_13/clip_by_valueMaximum-tf.clip_by_value_13/clip_by_value/Minimum:z:0,tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_13/clip_by_value?
#tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_9/FloorDiv/y?
!tf.compat.v1.floor_div_9/FloorDivFloorDiv%tf.clip_by_value_13/clip_by_value:z:0,tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_9/FloorDiv?
%tf.math.greater_equal_13/GreaterEqualGreaterEqual"flatten_9/PartitionedCall:output:0'tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_13/GreaterEqual?
tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_9/FloorMod/y?
tf.math.floormod_9/FloorModFloorMod%tf.clip_by_value_13/clip_by_value:z:0&tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_9/FloorMod?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_13/clip_by_value:z:0embedding_29_300118953*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_29_layer_call_and_return_conditional_losses_3001188392&
$embedding_29/StatefulPartitionedCall?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_9/FloorDiv:z:0embedding_27_300118956*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_27_layer_call_and_return_conditional_losses_3001188612&
$embedding_27/StatefulPartitionedCall?
tf.cast_13/CastCast)tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_13/Cast?
tf.__operators__.add_26/AddV2AddV2-embedding_29/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_26/AddV2?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_9/FloorMod:z:0embedding_28_300118961*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_28_layer_call_and_return_conditional_losses_3001188852&
$embedding_28/StatefulPartitionedCall?
tf.__operators__.add_27/AddV2AddV2!tf.__operators__.add_26/AddV2:z:0-embedding_28/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_27/AddV2?
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_9/ExpandDims/dim?
tf.expand_dims_9/ExpandDims
ExpandDimstf.cast_13/Cast:y:0(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_9/ExpandDims?
tf.math.multiply_9/MulMul!tf.__operators__.add_27/AddV2:z:0$tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_9/Mul?
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_9/Sum/reduction_indices?
tf.math.reduce_sum_9/SumSumtf.math.multiply_9/Mul:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_9/Sum?
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?9
?
F__inference_model_9_layer_call_and_return_conditional_losses_300120516

inputs+
'tf_math_greater_equal_13_greaterequal_y+
'embedding_29_embedding_lookup_300120490+
'embedding_27_embedding_lookup_300120496+
'embedding_28_embedding_lookup_300120504
identity??embedding_27/embedding_lookup?embedding_28/embedding_lookup?embedding_29/embedding_lookups
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_9/Const?
flatten_9/ReshapeReshapeinputsflatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_9/Reshape?
+tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_13/clip_by_value/Minimum/y?
)tf.clip_by_value_13/clip_by_value/MinimumMinimumflatten_9/Reshape:output:04tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_13/clip_by_value/Minimum?
#tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_13/clip_by_value/y?
!tf.clip_by_value_13/clip_by_valueMaximum-tf.clip_by_value_13/clip_by_value/Minimum:z:0,tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_13/clip_by_value?
#tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_9/FloorDiv/y?
!tf.compat.v1.floor_div_9/FloorDivFloorDiv%tf.clip_by_value_13/clip_by_value:z:0,tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_9/FloorDiv?
%tf.math.greater_equal_13/GreaterEqualGreaterEqualflatten_9/Reshape:output:0'tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_13/GreaterEqual?
tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_9/FloorMod/y?
tf.math.floormod_9/FloorModFloorMod%tf.clip_by_value_13/clip_by_value:z:0&tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_9/FloorMod?
embedding_29/CastCast%tf.clip_by_value_13/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_29/Cast?
embedding_29/embedding_lookupResourceGather'embedding_29_embedding_lookup_300120490embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_29/embedding_lookup/300120490*,
_output_shapes
:??????????*
dtype02
embedding_29/embedding_lookup?
&embedding_29/embedding_lookup/IdentityIdentity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_29/embedding_lookup/300120490*,
_output_shapes
:??????????2(
&embedding_29/embedding_lookup/Identity?
(embedding_29/embedding_lookup/Identity_1Identity/embedding_29/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_29/embedding_lookup/Identity_1?
embedding_27/CastCast%tf.compat.v1.floor_div_9/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_27/Cast?
embedding_27/embedding_lookupResourceGather'embedding_27_embedding_lookup_300120496embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_27/embedding_lookup/300120496*,
_output_shapes
:??????????*
dtype02
embedding_27/embedding_lookup?
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_27/embedding_lookup/300120496*,
_output_shapes
:??????????2(
&embedding_27/embedding_lookup/Identity?
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_27/embedding_lookup/Identity_1?
tf.cast_13/CastCast)tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_13/Cast?
tf.__operators__.add_26/AddV2AddV21embedding_29/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_26/AddV2?
embedding_28/CastCasttf.math.floormod_9/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_28/Cast?
embedding_28/embedding_lookupResourceGather'embedding_28_embedding_lookup_300120504embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_28/embedding_lookup/300120504*,
_output_shapes
:??????????*
dtype02
embedding_28/embedding_lookup?
&embedding_28/embedding_lookup/IdentityIdentity&embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_28/embedding_lookup/300120504*,
_output_shapes
:??????????2(
&embedding_28/embedding_lookup/Identity?
(embedding_28/embedding_lookup/Identity_1Identity/embedding_28/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_28/embedding_lookup/Identity_1?
tf.__operators__.add_27/AddV2AddV2!tf.__operators__.add_26/AddV2:z:01embedding_28/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_27/AddV2?
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_9/ExpandDims/dim?
tf.expand_dims_9/ExpandDims
ExpandDimstf.cast_13/Cast:y:0(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_9/ExpandDims?
tf.math.multiply_9/MulMul!tf.__operators__.add_27/AddV2:z:0$tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_9/Mul?
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_9/Sum/reduction_indices?
tf.math.reduce_sum_9/SumSumtf.math.multiply_9/Mul:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_9/Sum?
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^embedding_27/embedding_lookup^embedding_28/embedding_lookup^embedding_29/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_27/embedding_lookupembedding_27/embedding_lookup2>
embedding_28/embedding_lookupembedding_28/embedding_lookup2>
embedding_29/embedding_lookupembedding_29/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
K__inference_embedding_26_layer_call_and_return_conditional_losses_300118613

inputs
embedding_lookup_300118607
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300118607Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300118607*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300118607*,
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
?
?
,__inference_dense_40_layer_call_fn_300120682

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_40_layer_call_and_return_conditional_losses_3001192312
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
?
d
H__inference_flatten_9_layer_call_and_return_conditional_losses_300120852

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
I
-__inference_flatten_9_layer_call_fn_300120857

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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_9_layer_call_and_return_conditional_losses_3001188112
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
?
?
+__inference_model_8_layer_call_fn_300118801
input_9
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187902
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
_user_specified_name	input_9:

_output_shapes
: 
?	
?
G__inference_dense_44_layer_call_and_return_conditional_losses_300120775

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
K__inference_embedding_24_layer_call_and_return_conditional_losses_300120822

inputs
embedding_lookup_300120816
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300120816Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300120816*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300120816*,
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
?
?
2__inference_custom_model_4_layer_call_fn_300120295

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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_4_layer_call_and_return_conditional_losses_3001195812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
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
?
,__inference_dense_38_layer_call_fn_300120663

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_38_layer_call_and_return_conditional_losses_3001192052
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

/__inference_normalize_4_layer_call_fn_300120765
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
GPU2 *0J 8? *S
fNRL
J__inference_normalize_4_layer_call_and_return_conditional_losses_3001193492
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
?	
?
G__inference_dense_39_layer_call_and_return_conditional_losses_300119151

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_model_9_layer_call_fn_300118982
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001189712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:

_output_shapes
: 
?-
?
F__inference_model_9_layer_call_and_return_conditional_losses_300118936
input_10+
'tf_math_greater_equal_13_greaterequal_y
embedding_29_300118918
embedding_27_300118921
embedding_28_300118926
identity??$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?
flatten_9/PartitionedCallPartitionedCallinput_10*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_9_layer_call_and_return_conditional_losses_3001188112
flatten_9/PartitionedCall?
+tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_13/clip_by_value/Minimum/y?
)tf.clip_by_value_13/clip_by_value/MinimumMinimum"flatten_9/PartitionedCall:output:04tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_13/clip_by_value/Minimum?
#tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_13/clip_by_value/y?
!tf.clip_by_value_13/clip_by_valueMaximum-tf.clip_by_value_13/clip_by_value/Minimum:z:0,tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_13/clip_by_value?
#tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_9/FloorDiv/y?
!tf.compat.v1.floor_div_9/FloorDivFloorDiv%tf.clip_by_value_13/clip_by_value:z:0,tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_9/FloorDiv?
%tf.math.greater_equal_13/GreaterEqualGreaterEqual"flatten_9/PartitionedCall:output:0'tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_13/GreaterEqual?
tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_9/FloorMod/y?
tf.math.floormod_9/FloorModFloorMod%tf.clip_by_value_13/clip_by_value:z:0&tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_9/FloorMod?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_13/clip_by_value:z:0embedding_29_300118918*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_29_layer_call_and_return_conditional_losses_3001188392&
$embedding_29/StatefulPartitionedCall?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_9/FloorDiv:z:0embedding_27_300118921*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_27_layer_call_and_return_conditional_losses_3001188612&
$embedding_27/StatefulPartitionedCall?
tf.cast_13/CastCast)tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_13/Cast?
tf.__operators__.add_26/AddV2AddV2-embedding_29/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_26/AddV2?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_9/FloorMod:z:0embedding_28_300118926*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_28_layer_call_and_return_conditional_losses_3001188852&
$embedding_28/StatefulPartitionedCall?
tf.__operators__.add_27/AddV2AddV2!tf.__operators__.add_26/AddV2:z:0-embedding_28/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_27/AddV2?
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_9/ExpandDims/dim?
tf.expand_dims_9/ExpandDims
ExpandDimstf.cast_13/Cast:y:0(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_9/ExpandDims?
tf.math.multiply_9/MulMul!tf.__operators__.add_27/AddV2:z:0$tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_9/Mul?
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_9/Sum/reduction_indices?
tf.math.reduce_sum_9/SumSumtf.math.multiply_9/Mul:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_9/Sum?
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:

_output_shapes
: 
?	
?
K__inference_embedding_25_layer_call_and_return_conditional_losses_300120839

inputs
embedding_lookup_300120833
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300120833Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300120833*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300120833*,
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
?
?
2__inference_custom_model_4_layer_call_fn_300119646

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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_4_layer_call_and_return_conditional_losses_3001195812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
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
?
?
2__inference_custom_model_4_layer_call_fn_300119807

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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_4_layer_call_and_return_conditional_losses_3001197422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
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
?Y
?

M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119484

cards0

cards1
bets+
'tf_math_greater_equal_14_greaterequal_y
model_8_300119399
model_8_300119401
model_8_300119403
model_8_300119405
model_9_300119408
model_9_300119410
model_9_300119412
model_9_300119414/
+tf_clip_by_value_14_clip_by_value_minimum_y'
#tf_clip_by_value_14_clip_by_value_y
dense_36_300119426
dense_36_300119428
dense_39_300119431
dense_39_300119433
dense_37_300119436
dense_37_300119438
dense_38_300119441
dense_38_300119443
dense_40_300119446
dense_40_300119448
dense_41_300119453
dense_41_300119455
dense_42_300119459
dense_42_300119461
dense_43_300119466
dense_43_300119468
normalize_4_300119473
normalize_4_300119475
dense_44_300119478
dense_44_300119480
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?model_8/StatefulPartitionedCall?model_9/StatefulPartitionedCall?#normalize_4/StatefulPartitionedCall?
%tf.math.greater_equal_14/GreaterEqualGreaterEqualbets'tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_14/GreaterEqual?
model_8/StatefulPartitionedCallStatefulPartitionedCallcards0model_8_300119399model_8_300119401model_8_300119403model_8_300119405*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187902!
model_8/StatefulPartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCallcards1model_9_300119408model_9_300119410model_9_300119412model_9_300119414*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001190162!
model_9/StatefulPartitionedCall?
)tf.clip_by_value_14/clip_by_value/MinimumMinimumbets+tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_14/clip_by_value/Minimum?
!tf.clip_by_value_14/clip_by_valueMaximum-tf.clip_by_value_14/clip_by_value/Minimum:z:0#tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_14/clip_by_value?
tf.cast_14/CastCast)tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_14/Castv
tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_12/concat/axis?
tf.concat_12/concatConcatV2(model_8/StatefulPartitionedCall:output:0(model_9/StatefulPartitionedCall:output:0!tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_12/concat
tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_13/concat/axis?
tf.concat_13/concatConcatV2%tf.clip_by_value_14/clip_by_value:z:0tf.cast_14/Cast:y:0!tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_13/concat?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalltf.concat_12/concat:output:0dense_36_300119426dense_36_300119428*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_36_layer_call_and_return_conditional_losses_3001191252"
 dense_36/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCalltf.concat_13/concat:output:0dense_39_300119431dense_39_300119433*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_39_layer_call_and_return_conditional_losses_3001191512"
 dense_39/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_300119436dense_37_300119438*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_37_layer_call_and_return_conditional_losses_3001191782"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_300119441dense_38_300119443*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_38_layer_call_and_return_conditional_losses_3001192052"
 dense_38/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_300119446dense_40_300119448*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_40_layer_call_and_return_conditional_losses_3001192312"
 dense_40/StatefulPartitionedCall
tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_14/concat/axis?
tf.concat_14/concatConcatV2)dense_38/StatefulPartitionedCall:output:0)dense_40/StatefulPartitionedCall:output:0!tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_14/concat?
 dense_41/StatefulPartitionedCallStatefulPartitionedCalltf.concat_14/concat:output:0dense_41_300119453dense_41_300119455*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_41_layer_call_and_return_conditional_losses_3001192592"
 dense_41/StatefulPartitionedCall?
tf.nn.relu_12/ReluRelu)dense_41/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_12/Relu?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_12/Relu:activations:0dense_42_300119459dense_42_300119461*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_42_layer_call_and_return_conditional_losses_3001192862"
 dense_42/StatefulPartitionedCall?
tf.__operators__.add_28/AddV2AddV2)dense_42/StatefulPartitionedCall:output:0 tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_28/AddV2?
tf.nn.relu_13/ReluRelu!tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_13/Relu?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_13/Relu:activations:0dense_43_300119466dense_43_300119468*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_43_layer_call_and_return_conditional_losses_3001193142"
 dense_43/StatefulPartitionedCall?
tf.__operators__.add_29/AddV2AddV2)dense_43/StatefulPartitionedCall:output:0 tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_29/AddV2?
tf.nn.relu_14/ReluRelu!tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_14/Relu?
#normalize_4/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_14/Relu:activations:0normalize_4_300119473normalize_4_300119475*
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
GPU2 *0J 8? *S
fNRL
J__inference_normalize_4_layer_call_and_return_conditional_losses_3001193492%
#normalize_4/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall,normalize_4/StatefulPartitionedCall:output:0dense_44_300119478dense_44_300119480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_44_layer_call_and_return_conditional_losses_3001193752"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall ^model_8/StatefulPartitionedCall ^model_9/StatefulPartitionedCall$^normalize_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall2J
#normalize_4/StatefulPartitionedCall#normalize_4/StatefulPartitionedCall:O K
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
?9
?
F__inference_model_9_layer_call_and_return_conditional_losses_300120558

inputs+
'tf_math_greater_equal_13_greaterequal_y+
'embedding_29_embedding_lookup_300120532+
'embedding_27_embedding_lookup_300120538+
'embedding_28_embedding_lookup_300120546
identity??embedding_27/embedding_lookup?embedding_28/embedding_lookup?embedding_29/embedding_lookups
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_9/Const?
flatten_9/ReshapeReshapeinputsflatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_9/Reshape?
+tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_13/clip_by_value/Minimum/y?
)tf.clip_by_value_13/clip_by_value/MinimumMinimumflatten_9/Reshape:output:04tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_13/clip_by_value/Minimum?
#tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_13/clip_by_value/y?
!tf.clip_by_value_13/clip_by_valueMaximum-tf.clip_by_value_13/clip_by_value/Minimum:z:0,tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_13/clip_by_value?
#tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_9/FloorDiv/y?
!tf.compat.v1.floor_div_9/FloorDivFloorDiv%tf.clip_by_value_13/clip_by_value:z:0,tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_9/FloorDiv?
%tf.math.greater_equal_13/GreaterEqualGreaterEqualflatten_9/Reshape:output:0'tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_13/GreaterEqual?
tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_9/FloorMod/y?
tf.math.floormod_9/FloorModFloorMod%tf.clip_by_value_13/clip_by_value:z:0&tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_9/FloorMod?
embedding_29/CastCast%tf.clip_by_value_13/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_29/Cast?
embedding_29/embedding_lookupResourceGather'embedding_29_embedding_lookup_300120532embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_29/embedding_lookup/300120532*,
_output_shapes
:??????????*
dtype02
embedding_29/embedding_lookup?
&embedding_29/embedding_lookup/IdentityIdentity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_29/embedding_lookup/300120532*,
_output_shapes
:??????????2(
&embedding_29/embedding_lookup/Identity?
(embedding_29/embedding_lookup/Identity_1Identity/embedding_29/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_29/embedding_lookup/Identity_1?
embedding_27/CastCast%tf.compat.v1.floor_div_9/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_27/Cast?
embedding_27/embedding_lookupResourceGather'embedding_27_embedding_lookup_300120538embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_27/embedding_lookup/300120538*,
_output_shapes
:??????????*
dtype02
embedding_27/embedding_lookup?
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_27/embedding_lookup/300120538*,
_output_shapes
:??????????2(
&embedding_27/embedding_lookup/Identity?
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_27/embedding_lookup/Identity_1?
tf.cast_13/CastCast)tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_13/Cast?
tf.__operators__.add_26/AddV2AddV21embedding_29/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_26/AddV2?
embedding_28/CastCasttf.math.floormod_9/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_28/Cast?
embedding_28/embedding_lookupResourceGather'embedding_28_embedding_lookup_300120546embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_28/embedding_lookup/300120546*,
_output_shapes
:??????????*
dtype02
embedding_28/embedding_lookup?
&embedding_28/embedding_lookup/IdentityIdentity&embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_28/embedding_lookup/300120546*,
_output_shapes
:??????????2(
&embedding_28/embedding_lookup/Identity?
(embedding_28/embedding_lookup/Identity_1Identity/embedding_28/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_28/embedding_lookup/Identity_1?
tf.__operators__.add_27/AddV2AddV2!tf.__operators__.add_26/AddV2:z:01embedding_28/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_27/AddV2?
tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_9/ExpandDims/dim?
tf.expand_dims_9/ExpandDims
ExpandDimstf.cast_13/Cast:y:0(tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_9/ExpandDims?
tf.math.multiply_9/MulMul!tf.__operators__.add_27/AddV2:z:0$tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_9/Mul?
*tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_9/Sum/reduction_indices?
tf.math.reduce_sum_9/SumSumtf.math.multiply_9/Mul:z:03tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_9/Sum?
IdentityIdentity!tf.math.reduce_sum_9/Sum:output:0^embedding_27/embedding_lookup^embedding_28/embedding_lookup^embedding_29/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_27/embedding_lookupembedding_27/embedding_lookup2>
embedding_28/embedding_lookupembedding_28/embedding_lookup2>
embedding_29/embedding_lookupembedding_29/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
,__inference_dense_43_layer_call_fn_300120739

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_43_layer_call_and_return_conditional_losses_3001193142
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
G__inference_dense_37_layer_call_and_return_conditional_losses_300120615

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
??
?,
%__inference__traced_restore_300121440
file_prefix$
 assignvariableop_dense_36_kernel$
 assignvariableop_1_dense_36_bias&
"assignvariableop_2_dense_37_kernel$
 assignvariableop_3_dense_37_bias&
"assignvariableop_4_dense_39_kernel$
 assignvariableop_5_dense_39_bias&
"assignvariableop_6_dense_38_kernel$
 assignvariableop_7_dense_38_bias&
"assignvariableop_8_dense_40_kernel$
 assignvariableop_9_dense_40_bias'
#assignvariableop_10_dense_41_kernel%
!assignvariableop_11_dense_41_bias'
#assignvariableop_12_dense_42_kernel%
!assignvariableop_13_dense_42_bias'
#assignvariableop_14_dense_43_kernel%
!assignvariableop_15_dense_43_bias'
#assignvariableop_16_dense_44_kernel%
!assignvariableop_17_dense_44_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate/
+assignvariableop_23_embedding_26_embeddings/
+assignvariableop_24_embedding_24_embeddings/
+assignvariableop_25_embedding_25_embeddings/
+assignvariableop_26_embedding_29_embeddings/
+assignvariableop_27_embedding_27_embeddings/
+assignvariableop_28_embedding_28_embeddings8
4assignvariableop_29_normalize_4_normalization_4_mean<
8assignvariableop_30_normalize_4_normalization_4_variance9
5assignvariableop_31_normalize_4_normalization_4_count
assignvariableop_32_total
assignvariableop_33_count.
*assignvariableop_34_adam_dense_36_kernel_m,
(assignvariableop_35_adam_dense_36_bias_m.
*assignvariableop_36_adam_dense_37_kernel_m,
(assignvariableop_37_adam_dense_37_bias_m.
*assignvariableop_38_adam_dense_39_kernel_m,
(assignvariableop_39_adam_dense_39_bias_m.
*assignvariableop_40_adam_dense_38_kernel_m,
(assignvariableop_41_adam_dense_38_bias_m.
*assignvariableop_42_adam_dense_40_kernel_m,
(assignvariableop_43_adam_dense_40_bias_m.
*assignvariableop_44_adam_dense_41_kernel_m,
(assignvariableop_45_adam_dense_41_bias_m.
*assignvariableop_46_adam_dense_42_kernel_m,
(assignvariableop_47_adam_dense_42_bias_m.
*assignvariableop_48_adam_dense_43_kernel_m,
(assignvariableop_49_adam_dense_43_bias_m.
*assignvariableop_50_adam_dense_44_kernel_m,
(assignvariableop_51_adam_dense_44_bias_m6
2assignvariableop_52_adam_embedding_26_embeddings_m6
2assignvariableop_53_adam_embedding_24_embeddings_m6
2assignvariableop_54_adam_embedding_25_embeddings_m6
2assignvariableop_55_adam_embedding_29_embeddings_m6
2assignvariableop_56_adam_embedding_27_embeddings_m6
2assignvariableop_57_adam_embedding_28_embeddings_m.
*assignvariableop_58_adam_dense_36_kernel_v,
(assignvariableop_59_adam_dense_36_bias_v.
*assignvariableop_60_adam_dense_37_kernel_v,
(assignvariableop_61_adam_dense_37_bias_v.
*assignvariableop_62_adam_dense_39_kernel_v,
(assignvariableop_63_adam_dense_39_bias_v.
*assignvariableop_64_adam_dense_38_kernel_v,
(assignvariableop_65_adam_dense_38_bias_v.
*assignvariableop_66_adam_dense_40_kernel_v,
(assignvariableop_67_adam_dense_40_bias_v.
*assignvariableop_68_adam_dense_41_kernel_v,
(assignvariableop_69_adam_dense_41_bias_v.
*assignvariableop_70_adam_dense_42_kernel_v,
(assignvariableop_71_adam_dense_42_bias_v.
*assignvariableop_72_adam_dense_43_kernel_v,
(assignvariableop_73_adam_dense_43_bias_v.
*assignvariableop_74_adam_dense_44_kernel_v,
(assignvariableop_75_adam_dense_44_bias_v6
2assignvariableop_76_adam_embedding_26_embeddings_v6
2assignvariableop_77_adam_embedding_24_embeddings_v6
2assignvariableop_78_adam_embedding_25_embeddings_v6
2assignvariableop_79_adam_embedding_29_embeddings_v6
2assignvariableop_80_adam_embedding_27_embeddings_v6
2assignvariableop_81_adam_embedding_28_embeddings_v
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?+
value?+B?+SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_36_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_36_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_37_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_37_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_39_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_39_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_38_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_38_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_40_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_40_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_41_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_41_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_42_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_42_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_43_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_43_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_44_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_44_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_26_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_embedding_24_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_embedding_25_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_embedding_29_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_embedding_27_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_embedding_28_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_normalize_4_normalization_4_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp8assignvariableop_30_normalize_4_normalization_4_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_normalize_4_normalization_4_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_36_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_36_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_37_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_37_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_39_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_39_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_38_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_38_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_40_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_40_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_41_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_41_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_42_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_42_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_43_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_43_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_44_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_44_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_embedding_26_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_embedding_24_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_embedding_25_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_embedding_29_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_embedding_27_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_embedding_28_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_36_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_36_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_dense_37_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_dense_37_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_39_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_39_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_38_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_38_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dense_40_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_dense_40_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dense_41_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dense_41_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_42_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_42_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_43_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_43_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_dense_44_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dense_44_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_embedding_26_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp2assignvariableop_77_adam_embedding_24_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_embedding_25_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_embedding_29_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_embedding_27_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_embedding_28_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82?
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
G__inference_dense_42_layer_call_and_return_conditional_losses_300120711

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
G__inference_dense_38_layer_call_and_return_conditional_losses_300120654

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
?
I
-__inference_flatten_8_layer_call_fn_300120795

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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_3001185852
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
?-
?
F__inference_model_8_layer_call_and_return_conditional_losses_300118678
input_9+
'tf_math_greater_equal_12_greaterequal_y
embedding_26_300118622
embedding_24_300118644
embedding_25_300118668
identity??$embedding_24/StatefulPartitionedCall?$embedding_25/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinput_9*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_3001185852
flatten_8/PartitionedCall?
+tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_12/clip_by_value/Minimum/y?
)tf.clip_by_value_12/clip_by_value/MinimumMinimum"flatten_8/PartitionedCall:output:04tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_12/clip_by_value/Minimum?
#tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_12/clip_by_value/y?
!tf.clip_by_value_12/clip_by_valueMaximum-tf.clip_by_value_12/clip_by_value/Minimum:z:0,tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_12/clip_by_value?
#tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_8/FloorDiv/y?
!tf.compat.v1.floor_div_8/FloorDivFloorDiv%tf.clip_by_value_12/clip_by_value:z:0,tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_8/FloorDiv?
%tf.math.greater_equal_12/GreaterEqualGreaterEqual"flatten_8/PartitionedCall:output:0'tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_12/GreaterEqual?
tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_8/FloorMod/y?
tf.math.floormod_8/FloorModFloorMod%tf.clip_by_value_12/clip_by_value:z:0&tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_8/FloorMod?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_12/clip_by_value:z:0embedding_26_300118622*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_26_layer_call_and_return_conditional_losses_3001186132&
$embedding_26/StatefulPartitionedCall?
$embedding_24/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_8/FloorDiv:z:0embedding_24_300118644*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_24_layer_call_and_return_conditional_losses_3001186352&
$embedding_24/StatefulPartitionedCall?
tf.cast_12/CastCast)tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_12/Cast?
tf.__operators__.add_24/AddV2AddV2-embedding_26/StatefulPartitionedCall:output:0-embedding_24/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_24/AddV2?
$embedding_25/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_8/FloorMod:z:0embedding_25_300118668*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_25_layer_call_and_return_conditional_losses_3001186592&
$embedding_25/StatefulPartitionedCall?
tf.__operators__.add_25/AddV2AddV2!tf.__operators__.add_24/AddV2:z:0-embedding_25/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_25/AddV2?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimstf.cast_12/Cast:y:0(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_8/ExpandDims?
tf.math.multiply_8/MulMul!tf.__operators__.add_25/AddV2:z:0$tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_8/Mul?
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_8/Sum/reduction_indices?
tf.math.reduce_sum_8/SumSumtf.math.multiply_8/Mul:z:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_8/Sum?
IdentityIdentity!tf.math.reduce_sum_8/Sum:output:0%^embedding_24/StatefulPartitionedCall%^embedding_25/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_24/StatefulPartitionedCall$embedding_24/StatefulPartitionedCall2L
$embedding_25/StatefulPartitionedCall$embedding_25/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:

_output_shapes
: 
?9
?
F__inference_model_8_layer_call_and_return_conditional_losses_300120406

inputs+
'tf_math_greater_equal_12_greaterequal_y+
'embedding_26_embedding_lookup_300120380+
'embedding_24_embedding_lookup_300120386+
'embedding_25_embedding_lookup_300120394
identity??embedding_24/embedding_lookup?embedding_25/embedding_lookup?embedding_26/embedding_lookups
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshapeinputsflatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_8/Reshape?
+tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_12/clip_by_value/Minimum/y?
)tf.clip_by_value_12/clip_by_value/MinimumMinimumflatten_8/Reshape:output:04tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_12/clip_by_value/Minimum?
#tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_12/clip_by_value/y?
!tf.clip_by_value_12/clip_by_valueMaximum-tf.clip_by_value_12/clip_by_value/Minimum:z:0,tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_12/clip_by_value?
#tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_8/FloorDiv/y?
!tf.compat.v1.floor_div_8/FloorDivFloorDiv%tf.clip_by_value_12/clip_by_value:z:0,tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_8/FloorDiv?
%tf.math.greater_equal_12/GreaterEqualGreaterEqualflatten_8/Reshape:output:0'tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_12/GreaterEqual?
tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_8/FloorMod/y?
tf.math.floormod_8/FloorModFloorMod%tf.clip_by_value_12/clip_by_value:z:0&tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_8/FloorMod?
embedding_26/CastCast%tf.clip_by_value_12/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_26/Cast?
embedding_26/embedding_lookupResourceGather'embedding_26_embedding_lookup_300120380embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_26/embedding_lookup/300120380*,
_output_shapes
:??????????*
dtype02
embedding_26/embedding_lookup?
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_26/embedding_lookup/300120380*,
_output_shapes
:??????????2(
&embedding_26/embedding_lookup/Identity?
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_26/embedding_lookup/Identity_1?
embedding_24/CastCast%tf.compat.v1.floor_div_8/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_24/Cast?
embedding_24/embedding_lookupResourceGather'embedding_24_embedding_lookup_300120386embedding_24/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_24/embedding_lookup/300120386*,
_output_shapes
:??????????*
dtype02
embedding_24/embedding_lookup?
&embedding_24/embedding_lookup/IdentityIdentity&embedding_24/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_24/embedding_lookup/300120386*,
_output_shapes
:??????????2(
&embedding_24/embedding_lookup/Identity?
(embedding_24/embedding_lookup/Identity_1Identity/embedding_24/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_24/embedding_lookup/Identity_1?
tf.cast_12/CastCast)tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_12/Cast?
tf.__operators__.add_24/AddV2AddV21embedding_26/embedding_lookup/Identity_1:output:01embedding_24/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_24/AddV2?
embedding_25/CastCasttf.math.floormod_8/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_25/Cast?
embedding_25/embedding_lookupResourceGather'embedding_25_embedding_lookup_300120394embedding_25/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_25/embedding_lookup/300120394*,
_output_shapes
:??????????*
dtype02
embedding_25/embedding_lookup?
&embedding_25/embedding_lookup/IdentityIdentity&embedding_25/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_25/embedding_lookup/300120394*,
_output_shapes
:??????????2(
&embedding_25/embedding_lookup/Identity?
(embedding_25/embedding_lookup/Identity_1Identity/embedding_25/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_25/embedding_lookup/Identity_1?
tf.__operators__.add_25/AddV2AddV2!tf.__operators__.add_24/AddV2:z:01embedding_25/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_25/AddV2?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimstf.cast_12/Cast:y:0(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_8/ExpandDims?
tf.math.multiply_8/MulMul!tf.__operators__.add_25/AddV2:z:0$tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_8/Mul?
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_8/Sum/reduction_indices?
tf.math.reduce_sum_8/SumSumtf.math.multiply_8/Mul:z:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_8/Sum?
IdentityIdentity!tf.math.reduce_sum_8/Sum:output:0^embedding_24/embedding_lookup^embedding_25/embedding_lookup^embedding_26/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_24/embedding_lookupembedding_24/embedding_lookup2>
embedding_25/embedding_lookupembedding_25/embedding_lookup2>
embedding_26/embedding_lookupembedding_26/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?Y
?

M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119581

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_14_greaterequal_y
model_8_300119496
model_8_300119498
model_8_300119500
model_8_300119502
model_9_300119505
model_9_300119507
model_9_300119509
model_9_300119511/
+tf_clip_by_value_14_clip_by_value_minimum_y'
#tf_clip_by_value_14_clip_by_value_y
dense_36_300119523
dense_36_300119525
dense_39_300119528
dense_39_300119530
dense_37_300119533
dense_37_300119535
dense_38_300119538
dense_38_300119540
dense_40_300119543
dense_40_300119545
dense_41_300119550
dense_41_300119552
dense_42_300119556
dense_42_300119558
dense_43_300119563
dense_43_300119565
normalize_4_300119570
normalize_4_300119572
dense_44_300119575
dense_44_300119577
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?model_8/StatefulPartitionedCall?model_9/StatefulPartitionedCall?#normalize_4/StatefulPartitionedCall?
%tf.math.greater_equal_14/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_14/GreaterEqual?
model_8/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_8_300119496model_8_300119498model_8_300119500model_8_300119502*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187452!
model_8/StatefulPartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_9_300119505model_9_300119507model_9_300119509model_9_300119511*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001189712!
model_9/StatefulPartitionedCall?
)tf.clip_by_value_14/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_14/clip_by_value/Minimum?
!tf.clip_by_value_14/clip_by_valueMaximum-tf.clip_by_value_14/clip_by_value/Minimum:z:0#tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_14/clip_by_value?
tf.cast_14/CastCast)tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_14/Castv
tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_12/concat/axis?
tf.concat_12/concatConcatV2(model_8/StatefulPartitionedCall:output:0(model_9/StatefulPartitionedCall:output:0!tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_12/concat
tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_13/concat/axis?
tf.concat_13/concatConcatV2%tf.clip_by_value_14/clip_by_value:z:0tf.cast_14/Cast:y:0!tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_13/concat?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalltf.concat_12/concat:output:0dense_36_300119523dense_36_300119525*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_36_layer_call_and_return_conditional_losses_3001191252"
 dense_36/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCalltf.concat_13/concat:output:0dense_39_300119528dense_39_300119530*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_39_layer_call_and_return_conditional_losses_3001191512"
 dense_39/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_300119533dense_37_300119535*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_37_layer_call_and_return_conditional_losses_3001191782"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_300119538dense_38_300119540*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_38_layer_call_and_return_conditional_losses_3001192052"
 dense_38/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_300119543dense_40_300119545*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_40_layer_call_and_return_conditional_losses_3001192312"
 dense_40/StatefulPartitionedCall
tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_14/concat/axis?
tf.concat_14/concatConcatV2)dense_38/StatefulPartitionedCall:output:0)dense_40/StatefulPartitionedCall:output:0!tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_14/concat?
 dense_41/StatefulPartitionedCallStatefulPartitionedCalltf.concat_14/concat:output:0dense_41_300119550dense_41_300119552*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_41_layer_call_and_return_conditional_losses_3001192592"
 dense_41/StatefulPartitionedCall?
tf.nn.relu_12/ReluRelu)dense_41/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_12/Relu?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_12/Relu:activations:0dense_42_300119556dense_42_300119558*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_42_layer_call_and_return_conditional_losses_3001192862"
 dense_42/StatefulPartitionedCall?
tf.__operators__.add_28/AddV2AddV2)dense_42/StatefulPartitionedCall:output:0 tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_28/AddV2?
tf.nn.relu_13/ReluRelu!tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_13/Relu?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_13/Relu:activations:0dense_43_300119563dense_43_300119565*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_43_layer_call_and_return_conditional_losses_3001193142"
 dense_43/StatefulPartitionedCall?
tf.__operators__.add_29/AddV2AddV2)dense_43/StatefulPartitionedCall:output:0 tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_29/AddV2?
tf.nn.relu_14/ReluRelu!tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_14/Relu?
#normalize_4/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_14/Relu:activations:0normalize_4_300119570normalize_4_300119572*
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
GPU2 *0J 8? *S
fNRL
J__inference_normalize_4_layer_call_and_return_conditional_losses_3001193492%
#normalize_4/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall,normalize_4/StatefulPartitionedCall:output:0dense_44_300119575dense_44_300119577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_44_layer_call_and_return_conditional_losses_3001193752"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall ^model_8/StatefulPartitionedCall ^model_9/StatefulPartitionedCall$^normalize_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall2J
#normalize_4/StatefulPartitionedCall#normalize_4/StatefulPartitionedCall:O K
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
?	
?
K__inference_embedding_24_layer_call_and_return_conditional_losses_300118635

inputs
embedding_lookup_300118629
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300118629Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300118629*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300118629*,
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
G__inference_dense_40_layer_call_and_return_conditional_losses_300120673

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
K__inference_embedding_29_layer_call_and_return_conditional_losses_300118839

inputs
embedding_lookup_300118833
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300118833Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300118833*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300118833*,
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
?	
?
K__inference_embedding_28_layer_call_and_return_conditional_losses_300118885

inputs
embedding_lookup_300118879
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300118879Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300118879*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300118879*,
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
?	
?
G__inference_dense_36_layer_call_and_return_conditional_losses_300119125

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
G__inference_dense_38_layer_call_and_return_conditional_losses_300119205

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
G__inference_dense_44_layer_call_and_return_conditional_losses_300119375

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
?
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300120226

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_14_greaterequal_y3
/model_8_tf_math_greater_equal_12_greaterequal_y3
/model_8_embedding_26_embedding_lookup_3001200763
/model_8_embedding_24_embedding_lookup_3001200823
/model_8_embedding_25_embedding_lookup_3001200903
/model_9_tf_math_greater_equal_13_greaterequal_y3
/model_9_embedding_29_embedding_lookup_3001201143
/model_9_embedding_27_embedding_lookup_3001201203
/model_9_embedding_28_embedding_lookup_300120128/
+tf_clip_by_value_14_clip_by_value_minimum_y'
#tf_clip_by_value_14_clip_by_value_y+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource+
'dense_42_matmul_readvariableop_resource,
(dense_42_biasadd_readvariableop_resource+
'dense_43_matmul_readvariableop_resource,
(dense_43_biasadd_readvariableop_resource?
;normalize_4_normalization_4_reshape_readvariableop_resourceA
=normalize_4_normalization_4_reshape_1_readvariableop_resource+
'dense_44_matmul_readvariableop_resource,
(dense_44_biasadd_readvariableop_resource
identity??dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?dense_40/BiasAdd/ReadVariableOp?dense_40/MatMul/ReadVariableOp?dense_41/BiasAdd/ReadVariableOp?dense_41/MatMul/ReadVariableOp?dense_42/BiasAdd/ReadVariableOp?dense_42/MatMul/ReadVariableOp?dense_43/BiasAdd/ReadVariableOp?dense_43/MatMul/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?dense_44/MatMul/ReadVariableOp?%model_8/embedding_24/embedding_lookup?%model_8/embedding_25/embedding_lookup?%model_8/embedding_26/embedding_lookup?%model_9/embedding_27/embedding_lookup?%model_9/embedding_28/embedding_lookup?%model_9/embedding_29/embedding_lookup?2normalize_4/normalization_4/Reshape/ReadVariableOp?4normalize_4/normalization_4/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_14/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_14/GreaterEqual?
model_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_8/flatten_8/Const?
model_8/flatten_8/ReshapeReshape
inputs_0_0 model_8/flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
model_8/flatten_8/Reshape?
3model_8/tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_8/tf.clip_by_value_12/clip_by_value/Minimum/y?
1model_8/tf.clip_by_value_12/clip_by_value/MinimumMinimum"model_8/flatten_8/Reshape:output:0<model_8/tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_8/tf.clip_by_value_12/clip_by_value/Minimum?
+model_8/tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_8/tf.clip_by_value_12/clip_by_value/y?
)model_8/tf.clip_by_value_12/clip_by_valueMaximum5model_8/tf.clip_by_value_12/clip_by_value/Minimum:z:04model_8/tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_8/tf.clip_by_value_12/clip_by_value?
+model_8/tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_8/tf.compat.v1.floor_div_8/FloorDiv/y?
)model_8/tf.compat.v1.floor_div_8/FloorDivFloorDiv-model_8/tf.clip_by_value_12/clip_by_value:z:04model_8/tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_8/tf.compat.v1.floor_div_8/FloorDiv?
-model_8/tf.math.greater_equal_12/GreaterEqualGreaterEqual"model_8/flatten_8/Reshape:output:0/model_8_tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_8/tf.math.greater_equal_12/GreaterEqual?
%model_8/tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_8/tf.math.floormod_8/FloorMod/y?
#model_8/tf.math.floormod_8/FloorModFloorMod-model_8/tf.clip_by_value_12/clip_by_value:z:0.model_8/tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_8/tf.math.floormod_8/FloorMod?
model_8/embedding_26/CastCast-model_8/tf.clip_by_value_12/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_8/embedding_26/Cast?
%model_8/embedding_26/embedding_lookupResourceGather/model_8_embedding_26_embedding_lookup_300120076model_8/embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_8/embedding_26/embedding_lookup/300120076*,
_output_shapes
:??????????*
dtype02'
%model_8/embedding_26/embedding_lookup?
.model_8/embedding_26/embedding_lookup/IdentityIdentity.model_8/embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_8/embedding_26/embedding_lookup/300120076*,
_output_shapes
:??????????20
.model_8/embedding_26/embedding_lookup/Identity?
0model_8/embedding_26/embedding_lookup/Identity_1Identity7model_8/embedding_26/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_8/embedding_26/embedding_lookup/Identity_1?
model_8/embedding_24/CastCast-model_8/tf.compat.v1.floor_div_8/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_8/embedding_24/Cast?
%model_8/embedding_24/embedding_lookupResourceGather/model_8_embedding_24_embedding_lookup_300120082model_8/embedding_24/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_8/embedding_24/embedding_lookup/300120082*,
_output_shapes
:??????????*
dtype02'
%model_8/embedding_24/embedding_lookup?
.model_8/embedding_24/embedding_lookup/IdentityIdentity.model_8/embedding_24/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_8/embedding_24/embedding_lookup/300120082*,
_output_shapes
:??????????20
.model_8/embedding_24/embedding_lookup/Identity?
0model_8/embedding_24/embedding_lookup/Identity_1Identity7model_8/embedding_24/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_8/embedding_24/embedding_lookup/Identity_1?
model_8/tf.cast_12/CastCast1model_8/tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_8/tf.cast_12/Cast?
%model_8/tf.__operators__.add_24/AddV2AddV29model_8/embedding_26/embedding_lookup/Identity_1:output:09model_8/embedding_24/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_8/tf.__operators__.add_24/AddV2?
model_8/embedding_25/CastCast'model_8/tf.math.floormod_8/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_8/embedding_25/Cast?
%model_8/embedding_25/embedding_lookupResourceGather/model_8_embedding_25_embedding_lookup_300120090model_8/embedding_25/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_8/embedding_25/embedding_lookup/300120090*,
_output_shapes
:??????????*
dtype02'
%model_8/embedding_25/embedding_lookup?
.model_8/embedding_25/embedding_lookup/IdentityIdentity.model_8/embedding_25/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_8/embedding_25/embedding_lookup/300120090*,
_output_shapes
:??????????20
.model_8/embedding_25/embedding_lookup/Identity?
0model_8/embedding_25/embedding_lookup/Identity_1Identity7model_8/embedding_25/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_8/embedding_25/embedding_lookup/Identity_1?
%model_8/tf.__operators__.add_25/AddV2AddV2)model_8/tf.__operators__.add_24/AddV2:z:09model_8/embedding_25/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_8/tf.__operators__.add_25/AddV2?
'model_8/tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_8/tf.expand_dims_8/ExpandDims/dim?
#model_8/tf.expand_dims_8/ExpandDims
ExpandDimsmodel_8/tf.cast_12/Cast:y:00model_8/tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_8/tf.expand_dims_8/ExpandDims?
model_8/tf.math.multiply_8/MulMul)model_8/tf.__operators__.add_25/AddV2:z:0,model_8/tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_8/tf.math.multiply_8/Mul?
2model_8/tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_8/tf.math.reduce_sum_8/Sum/reduction_indices?
 model_8/tf.math.reduce_sum_8/SumSum"model_8/tf.math.multiply_8/Mul:z:0;model_8/tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_8/tf.math.reduce_sum_8/Sum?
model_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_9/flatten_9/Const?
model_9/flatten_9/ReshapeReshape
inputs_0_1 model_9/flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
model_9/flatten_9/Reshape?
3model_9/tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_9/tf.clip_by_value_13/clip_by_value/Minimum/y?
1model_9/tf.clip_by_value_13/clip_by_value/MinimumMinimum"model_9/flatten_9/Reshape:output:0<model_9/tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_9/tf.clip_by_value_13/clip_by_value/Minimum?
+model_9/tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_9/tf.clip_by_value_13/clip_by_value/y?
)model_9/tf.clip_by_value_13/clip_by_valueMaximum5model_9/tf.clip_by_value_13/clip_by_value/Minimum:z:04model_9/tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_9/tf.clip_by_value_13/clip_by_value?
+model_9/tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_9/tf.compat.v1.floor_div_9/FloorDiv/y?
)model_9/tf.compat.v1.floor_div_9/FloorDivFloorDiv-model_9/tf.clip_by_value_13/clip_by_value:z:04model_9/tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_9/tf.compat.v1.floor_div_9/FloorDiv?
-model_9/tf.math.greater_equal_13/GreaterEqualGreaterEqual"model_9/flatten_9/Reshape:output:0/model_9_tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_9/tf.math.greater_equal_13/GreaterEqual?
%model_9/tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_9/tf.math.floormod_9/FloorMod/y?
#model_9/tf.math.floormod_9/FloorModFloorMod-model_9/tf.clip_by_value_13/clip_by_value:z:0.model_9/tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_9/tf.math.floormod_9/FloorMod?
model_9/embedding_29/CastCast-model_9/tf.clip_by_value_13/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_9/embedding_29/Cast?
%model_9/embedding_29/embedding_lookupResourceGather/model_9_embedding_29_embedding_lookup_300120114model_9/embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_9/embedding_29/embedding_lookup/300120114*,
_output_shapes
:??????????*
dtype02'
%model_9/embedding_29/embedding_lookup?
.model_9/embedding_29/embedding_lookup/IdentityIdentity.model_9/embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_9/embedding_29/embedding_lookup/300120114*,
_output_shapes
:??????????20
.model_9/embedding_29/embedding_lookup/Identity?
0model_9/embedding_29/embedding_lookup/Identity_1Identity7model_9/embedding_29/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_9/embedding_29/embedding_lookup/Identity_1?
model_9/embedding_27/CastCast-model_9/tf.compat.v1.floor_div_9/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_9/embedding_27/Cast?
%model_9/embedding_27/embedding_lookupResourceGather/model_9_embedding_27_embedding_lookup_300120120model_9/embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_9/embedding_27/embedding_lookup/300120120*,
_output_shapes
:??????????*
dtype02'
%model_9/embedding_27/embedding_lookup?
.model_9/embedding_27/embedding_lookup/IdentityIdentity.model_9/embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_9/embedding_27/embedding_lookup/300120120*,
_output_shapes
:??????????20
.model_9/embedding_27/embedding_lookup/Identity?
0model_9/embedding_27/embedding_lookup/Identity_1Identity7model_9/embedding_27/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_9/embedding_27/embedding_lookup/Identity_1?
model_9/tf.cast_13/CastCast1model_9/tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_9/tf.cast_13/Cast?
%model_9/tf.__operators__.add_26/AddV2AddV29model_9/embedding_29/embedding_lookup/Identity_1:output:09model_9/embedding_27/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_9/tf.__operators__.add_26/AddV2?
model_9/embedding_28/CastCast'model_9/tf.math.floormod_9/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_9/embedding_28/Cast?
%model_9/embedding_28/embedding_lookupResourceGather/model_9_embedding_28_embedding_lookup_300120128model_9/embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_9/embedding_28/embedding_lookup/300120128*,
_output_shapes
:??????????*
dtype02'
%model_9/embedding_28/embedding_lookup?
.model_9/embedding_28/embedding_lookup/IdentityIdentity.model_9/embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_9/embedding_28/embedding_lookup/300120128*,
_output_shapes
:??????????20
.model_9/embedding_28/embedding_lookup/Identity?
0model_9/embedding_28/embedding_lookup/Identity_1Identity7model_9/embedding_28/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_9/embedding_28/embedding_lookup/Identity_1?
%model_9/tf.__operators__.add_27/AddV2AddV2)model_9/tf.__operators__.add_26/AddV2:z:09model_9/embedding_28/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_9/tf.__operators__.add_27/AddV2?
'model_9/tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_9/tf.expand_dims_9/ExpandDims/dim?
#model_9/tf.expand_dims_9/ExpandDims
ExpandDimsmodel_9/tf.cast_13/Cast:y:00model_9/tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_9/tf.expand_dims_9/ExpandDims?
model_9/tf.math.multiply_9/MulMul)model_9/tf.__operators__.add_27/AddV2:z:0,model_9/tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_9/tf.math.multiply_9/Mul?
2model_9/tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_9/tf.math.reduce_sum_9/Sum/reduction_indices?
 model_9/tf.math.reduce_sum_9/SumSum"model_9/tf.math.multiply_9/Mul:z:0;model_9/tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_9/tf.math.reduce_sum_9/Sum?
)tf.clip_by_value_14/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_14/clip_by_value/Minimum?
!tf.clip_by_value_14/clip_by_valueMaximum-tf.clip_by_value_14/clip_by_value/Minimum:z:0#tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_14/clip_by_value?
tf.cast_14/CastCast)tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_14/Castv
tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_12/concat/axis?
tf.concat_12/concatConcatV2)model_8/tf.math.reduce_sum_8/Sum:output:0)model_9/tf.math.reduce_sum_9/Sum:output:0!tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_12/concat
tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_13/concat/axis?
tf.concat_13/concatConcatV2%tf.clip_by_value_14/clip_by_value:z:0tf.cast_14/Cast:y:0!tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_13/concat?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMultf.concat_12/concat:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Relu?
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_39/MatMul/ReadVariableOp?
dense_39/MatMulMatMultf.concat_13/concat:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_39/MatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_39/BiasAdd?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Relu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_37/BiasAddt
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_37/Relu?
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_38/MatMul/ReadVariableOp?
dense_38/MatMulMatMuldense_37/Relu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_38/MatMul?
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_38/BiasAdd/ReadVariableOp?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_38/BiasAddt
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_38/Relu?
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_40/MatMul/ReadVariableOp?
dense_40/MatMulMatMuldense_39/BiasAdd:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_40/MatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_40/BiasAdd
tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_14/concat/axis?
tf.concat_14/concatConcatV2dense_38/Relu:activations:0dense_40/BiasAdd:output:0!tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_14/concat?
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_41/MatMul/ReadVariableOp?
dense_41/MatMulMatMultf.concat_14/concat:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/MatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/BiasAdd~
tf.nn.relu_12/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_12/Relu?
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_42/MatMul/ReadVariableOp?
dense_42/MatMulMatMul tf.nn.relu_12/Relu:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/MatMul?
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_42/BiasAdd/ReadVariableOp?
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/BiasAdd?
tf.__operators__.add_28/AddV2AddV2dense_42/BiasAdd:output:0 tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_28/AddV2?
tf.nn.relu_13/ReluRelu!tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_13/Relu?
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_43/MatMul/ReadVariableOp?
dense_43/MatMulMatMul tf.nn.relu_13/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/MatMul?
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_43/BiasAdd/ReadVariableOp?
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/BiasAdd?
tf.__operators__.add_29/AddV2AddV2dense_43/BiasAdd:output:0 tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_29/AddV2?
tf.nn.relu_14/ReluRelu!tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_14/Relu?
2normalize_4/normalization_4/Reshape/ReadVariableOpReadVariableOp;normalize_4_normalization_4_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_4/normalization_4/Reshape/ReadVariableOp?
)normalize_4/normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_4/normalization_4/Reshape/shape?
#normalize_4/normalization_4/ReshapeReshape:normalize_4/normalization_4/Reshape/ReadVariableOp:value:02normalize_4/normalization_4/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_4/normalization_4/Reshape?
4normalize_4/normalization_4/Reshape_1/ReadVariableOpReadVariableOp=normalize_4_normalization_4_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_4/normalization_4/Reshape_1/ReadVariableOp?
+normalize_4/normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_4/normalization_4/Reshape_1/shape?
%normalize_4/normalization_4/Reshape_1Reshape<normalize_4/normalization_4/Reshape_1/ReadVariableOp:value:04normalize_4/normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_4/normalization_4/Reshape_1?
normalize_4/normalization_4/subSub tf.nn.relu_14/Relu:activations:0,normalize_4/normalization_4/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_4/normalization_4/sub?
 normalize_4/normalization_4/SqrtSqrt.normalize_4/normalization_4/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_4/normalization_4/Sqrt?
%normalize_4/normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_4/normalization_4/Maximum/y?
#normalize_4/normalization_4/MaximumMaximum$normalize_4/normalization_4/Sqrt:y:0.normalize_4/normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_4/normalization_4/Maximum?
#normalize_4/normalization_4/truedivRealDiv#normalize_4/normalization_4/sub:z:0'normalize_4/normalization_4/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_4/normalization_4/truediv?
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_44/MatMul/ReadVariableOp?
dense_44/MatMulMatMul'normalize_4/normalization_4/truediv:z:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_44/MatMul?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_44/BiasAdd?
IdentityIdentitydense_44/BiasAdd:output:0 ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp&^model_8/embedding_24/embedding_lookup&^model_8/embedding_25/embedding_lookup&^model_8/embedding_26/embedding_lookup&^model_9/embedding_27/embedding_lookup&^model_9/embedding_28/embedding_lookup&^model_9/embedding_29/embedding_lookup3^normalize_4/normalization_4/Reshape/ReadVariableOp5^normalize_4/normalization_4/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2N
%model_8/embedding_24/embedding_lookup%model_8/embedding_24/embedding_lookup2N
%model_8/embedding_25/embedding_lookup%model_8/embedding_25/embedding_lookup2N
%model_8/embedding_26/embedding_lookup%model_8/embedding_26/embedding_lookup2N
%model_9/embedding_27/embedding_lookup%model_9/embedding_27/embedding_lookup2N
%model_9/embedding_28/embedding_lookup%model_9/embedding_28/embedding_lookup2N
%model_9/embedding_29/embedding_lookup%model_9/embedding_29/embedding_lookup2h
2normalize_4/normalization_4/Reshape/ReadVariableOp2normalize_4/normalization_4/Reshape/ReadVariableOp2l
4normalize_4/normalization_4/Reshape_1/ReadVariableOp4normalize_4/normalization_4/Reshape_1/ReadVariableOp:S O
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
K__inference_embedding_25_layer_call_and_return_conditional_losses_300118659

inputs
embedding_lookup_300118653
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300118653Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300118653*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300118653*,
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
G__inference_dense_42_layer_call_and_return_conditional_losses_300119286

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
?Y
?

M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119742

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_14_greaterequal_y
model_8_300119657
model_8_300119659
model_8_300119661
model_8_300119663
model_9_300119666
model_9_300119668
model_9_300119670
model_9_300119672/
+tf_clip_by_value_14_clip_by_value_minimum_y'
#tf_clip_by_value_14_clip_by_value_y
dense_36_300119684
dense_36_300119686
dense_39_300119689
dense_39_300119691
dense_37_300119694
dense_37_300119696
dense_38_300119699
dense_38_300119701
dense_40_300119704
dense_40_300119706
dense_41_300119711
dense_41_300119713
dense_42_300119717
dense_42_300119719
dense_43_300119724
dense_43_300119726
normalize_4_300119731
normalize_4_300119733
dense_44_300119736
dense_44_300119738
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall? dense_38/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall? dense_42/StatefulPartitionedCall? dense_43/StatefulPartitionedCall? dense_44/StatefulPartitionedCall?model_8/StatefulPartitionedCall?model_9/StatefulPartitionedCall?#normalize_4/StatefulPartitionedCall?
%tf.math.greater_equal_14/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_14/GreaterEqual?
model_8/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_8_300119657model_8_300119659model_8_300119661model_8_300119663*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187902!
model_8/StatefulPartitionedCall?
model_9/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_9_300119666model_9_300119668model_9_300119670model_9_300119672*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001190162!
model_9/StatefulPartitionedCall?
)tf.clip_by_value_14/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_14/clip_by_value/Minimum?
!tf.clip_by_value_14/clip_by_valueMaximum-tf.clip_by_value_14/clip_by_value/Minimum:z:0#tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_14/clip_by_value?
tf.cast_14/CastCast)tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_14/Castv
tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_12/concat/axis?
tf.concat_12/concatConcatV2(model_8/StatefulPartitionedCall:output:0(model_9/StatefulPartitionedCall:output:0!tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_12/concat
tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_13/concat/axis?
tf.concat_13/concatConcatV2%tf.clip_by_value_14/clip_by_value:z:0tf.cast_14/Cast:y:0!tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_13/concat?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalltf.concat_12/concat:output:0dense_36_300119684dense_36_300119686*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_36_layer_call_and_return_conditional_losses_3001191252"
 dense_36/StatefulPartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCalltf.concat_13/concat:output:0dense_39_300119689dense_39_300119691*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_39_layer_call_and_return_conditional_losses_3001191512"
 dense_39/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_300119694dense_37_300119696*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_37_layer_call_and_return_conditional_losses_3001191782"
 dense_37/StatefulPartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_300119699dense_38_300119701*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_38_layer_call_and_return_conditional_losses_3001192052"
 dense_38/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_300119704dense_40_300119706*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_40_layer_call_and_return_conditional_losses_3001192312"
 dense_40/StatefulPartitionedCall
tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_14/concat/axis?
tf.concat_14/concatConcatV2)dense_38/StatefulPartitionedCall:output:0)dense_40/StatefulPartitionedCall:output:0!tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_14/concat?
 dense_41/StatefulPartitionedCallStatefulPartitionedCalltf.concat_14/concat:output:0dense_41_300119711dense_41_300119713*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_41_layer_call_and_return_conditional_losses_3001192592"
 dense_41/StatefulPartitionedCall?
tf.nn.relu_12/ReluRelu)dense_41/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_12/Relu?
 dense_42/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_12/Relu:activations:0dense_42_300119717dense_42_300119719*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_42_layer_call_and_return_conditional_losses_3001192862"
 dense_42/StatefulPartitionedCall?
tf.__operators__.add_28/AddV2AddV2)dense_42/StatefulPartitionedCall:output:0 tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_28/AddV2?
tf.nn.relu_13/ReluRelu!tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_13/Relu?
 dense_43/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_13/Relu:activations:0dense_43_300119724dense_43_300119726*
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
GPU2 *0J 8? *P
fKRI
G__inference_dense_43_layer_call_and_return_conditional_losses_3001193142"
 dense_43/StatefulPartitionedCall?
tf.__operators__.add_29/AddV2AddV2)dense_43/StatefulPartitionedCall:output:0 tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_29/AddV2?
tf.nn.relu_14/ReluRelu!tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_14/Relu?
#normalize_4/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_14/Relu:activations:0normalize_4_300119731normalize_4_300119733*
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
GPU2 *0J 8? *S
fNRL
J__inference_normalize_4_layer_call_and_return_conditional_losses_3001193492%
#normalize_4/StatefulPartitionedCall?
 dense_44/StatefulPartitionedCallStatefulPartitionedCall,normalize_4/StatefulPartitionedCall:output:0dense_44_300119736dense_44_300119738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_44_layer_call_and_return_conditional_losses_3001193752"
 dense_44/StatefulPartitionedCall?
IdentityIdentity)dense_44/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall!^dense_44/StatefulPartitionedCall ^model_8/StatefulPartitionedCall ^model_9/StatefulPartitionedCall$^normalize_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2D
 dense_44/StatefulPartitionedCall dense_44/StatefulPartitionedCall2B
model_8/StatefulPartitionedCallmodel_8/StatefulPartitionedCall2B
model_9/StatefulPartitionedCallmodel_9/StatefulPartitionedCall2J
#normalize_4/StatefulPartitionedCall#normalize_4/StatefulPartitionedCall:O K
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
?
?
,__inference_dense_42_layer_call_fn_300120720

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_42_layer_call_and_return_conditional_losses_3001192862
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
+__inference_model_9_layer_call_fn_300119027
input_10
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_9_layer_call_and_return_conditional_losses_3001190162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:

_output_shapes
: 
?
?
,__inference_dense_39_layer_call_fn_300120643

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_39_layer_call_and_return_conditional_losses_3001191512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_37_layer_call_fn_300120624

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_37_layer_call_and_return_conditional_losses_3001191782
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
?-
?
F__inference_model_8_layer_call_and_return_conditional_losses_300118710
input_9+
'tf_math_greater_equal_12_greaterequal_y
embedding_26_300118692
embedding_24_300118695
embedding_25_300118700
identity??$embedding_24/StatefulPartitionedCall?$embedding_25/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinput_9*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_3001185852
flatten_8/PartitionedCall?
+tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_12/clip_by_value/Minimum/y?
)tf.clip_by_value_12/clip_by_value/MinimumMinimum"flatten_8/PartitionedCall:output:04tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_12/clip_by_value/Minimum?
#tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_12/clip_by_value/y?
!tf.clip_by_value_12/clip_by_valueMaximum-tf.clip_by_value_12/clip_by_value/Minimum:z:0,tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_12/clip_by_value?
#tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_8/FloorDiv/y?
!tf.compat.v1.floor_div_8/FloorDivFloorDiv%tf.clip_by_value_12/clip_by_value:z:0,tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_8/FloorDiv?
%tf.math.greater_equal_12/GreaterEqualGreaterEqual"flatten_8/PartitionedCall:output:0'tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_12/GreaterEqual?
tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_8/FloorMod/y?
tf.math.floormod_8/FloorModFloorMod%tf.clip_by_value_12/clip_by_value:z:0&tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_8/FloorMod?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_12/clip_by_value:z:0embedding_26_300118692*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_26_layer_call_and_return_conditional_losses_3001186132&
$embedding_26/StatefulPartitionedCall?
$embedding_24/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_8/FloorDiv:z:0embedding_24_300118695*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_24_layer_call_and_return_conditional_losses_3001186352&
$embedding_24/StatefulPartitionedCall?
tf.cast_12/CastCast)tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_12/Cast?
tf.__operators__.add_24/AddV2AddV2-embedding_26/StatefulPartitionedCall:output:0-embedding_24/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_24/AddV2?
$embedding_25/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_8/FloorMod:z:0embedding_25_300118700*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_25_layer_call_and_return_conditional_losses_3001186592&
$embedding_25/StatefulPartitionedCall?
tf.__operators__.add_25/AddV2AddV2!tf.__operators__.add_24/AddV2:z:0-embedding_25/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_25/AddV2?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimstf.cast_12/Cast:y:0(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_8/ExpandDims?
tf.math.multiply_8/MulMul!tf.__operators__.add_25/AddV2:z:0$tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_8/Mul?
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_8/Sum/reduction_indices?
tf.math.reduce_sum_8/SumSumtf.math.multiply_8/Mul:z:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_8/Sum?
IdentityIdentity!tf.math.reduce_sum_8/Sum:output:0%^embedding_24/StatefulPartitionedCall%^embedding_25/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_24/StatefulPartitionedCall$embedding_24/StatefulPartitionedCall2L
$embedding_25/StatefulPartitionedCall$embedding_25/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_9:

_output_shapes
: 
?
?
'__inference_signature_wrapper_300119886
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
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference__wrapped_model_3001185752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
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
?
d
H__inference_flatten_9_layer_call_and_return_conditional_losses_300118811

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
?
?
J__inference_normalize_4_layer_call_and_return_conditional_losses_300119349
x3
/normalization_4_reshape_readvariableop_resource5
1normalization_4_reshape_1_readvariableop_resource
identity??&normalization_4/Reshape/ReadVariableOp?(normalization_4/Reshape_1/ReadVariableOp?
&normalization_4/Reshape/ReadVariableOpReadVariableOp/normalization_4_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_4/Reshape/ReadVariableOp?
normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_4/Reshape/shape?
normalization_4/ReshapeReshape.normalization_4/Reshape/ReadVariableOp:value:0&normalization_4/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_4/Reshape?
(normalization_4/Reshape_1/ReadVariableOpReadVariableOp1normalization_4_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_4/Reshape_1/ReadVariableOp?
normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_4/Reshape_1/shape?
normalization_4/Reshape_1Reshape0normalization_4/Reshape_1/ReadVariableOp:value:0(normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_4/Reshape_1?
normalization_4/subSubx normalization_4/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_4/sub?
normalization_4/SqrtSqrt"normalization_4/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_4/Sqrt{
normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_4/Maximum/y?
normalization_4/MaximumMaximumnormalization_4/Sqrt:y:0"normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_4/Maximum?
normalization_4/truedivRealDivnormalization_4/sub:z:0normalization_4/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_4/truediv?
IdentityIdentitynormalization_4/truediv:z:0'^normalization_4/Reshape/ReadVariableOp)^normalization_4/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_4/Reshape/ReadVariableOp&normalization_4/Reshape/ReadVariableOp2T
(normalization_4/Reshape_1/ReadVariableOp(normalization_4/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
v
0__inference_embedding_26_layer_call_fn_300120812

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
GPU2 *0J 8? *T
fORM
K__inference_embedding_26_layer_call_and_return_conditional_losses_3001186132
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
?
,__inference_dense_44_layer_call_fn_300120784

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
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_44_layer_call_and_return_conditional_losses_3001193752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
v
0__inference_embedding_29_layer_call_fn_300120874

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
GPU2 *0J 8? *T
fORM
K__inference_embedding_29_layer_call_and_return_conditional_losses_3001188392
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
v
0__inference_embedding_24_layer_call_fn_300120829

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
GPU2 *0J 8? *T
fORM
K__inference_embedding_24_layer_call_and_return_conditional_losses_3001186352
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
?-
?
F__inference_model_8_layer_call_and_return_conditional_losses_300118745

inputs+
'tf_math_greater_equal_12_greaterequal_y
embedding_26_300118727
embedding_24_300118730
embedding_25_300118735
identity??$embedding_24/StatefulPartitionedCall?$embedding_25/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCallinputs*
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
GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_8_layer_call_and_return_conditional_losses_3001185852
flatten_8/PartitionedCall?
+tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_12/clip_by_value/Minimum/y?
)tf.clip_by_value_12/clip_by_value/MinimumMinimum"flatten_8/PartitionedCall:output:04tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_12/clip_by_value/Minimum?
#tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_12/clip_by_value/y?
!tf.clip_by_value_12/clip_by_valueMaximum-tf.clip_by_value_12/clip_by_value/Minimum:z:0,tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_12/clip_by_value?
#tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_8/FloorDiv/y?
!tf.compat.v1.floor_div_8/FloorDivFloorDiv%tf.clip_by_value_12/clip_by_value:z:0,tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_8/FloorDiv?
%tf.math.greater_equal_12/GreaterEqualGreaterEqual"flatten_8/PartitionedCall:output:0'tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_12/GreaterEqual?
tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_8/FloorMod/y?
tf.math.floormod_8/FloorModFloorMod%tf.clip_by_value_12/clip_by_value:z:0&tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_8/FloorMod?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_12/clip_by_value:z:0embedding_26_300118727*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_26_layer_call_and_return_conditional_losses_3001186132&
$embedding_26/StatefulPartitionedCall?
$embedding_24/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_8/FloorDiv:z:0embedding_24_300118730*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_24_layer_call_and_return_conditional_losses_3001186352&
$embedding_24/StatefulPartitionedCall?
tf.cast_12/CastCast)tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_12/Cast?
tf.__operators__.add_24/AddV2AddV2-embedding_26/StatefulPartitionedCall:output:0-embedding_24/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_24/AddV2?
$embedding_25/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_8/FloorMod:z:0embedding_25_300118735*
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
GPU2 *0J 8? *T
fORM
K__inference_embedding_25_layer_call_and_return_conditional_losses_3001186592&
$embedding_25/StatefulPartitionedCall?
tf.__operators__.add_25/AddV2AddV2!tf.__operators__.add_24/AddV2:z:0-embedding_25/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_25/AddV2?
tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_8/ExpandDims/dim?
tf.expand_dims_8/ExpandDims
ExpandDimstf.cast_12/Cast:y:0(tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_8/ExpandDims?
tf.math.multiply_8/MulMul!tf.__operators__.add_25/AddV2:z:0$tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_8/Mul?
*tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_8/Sum/reduction_indices?
tf.math.reduce_sum_8/SumSumtf.math.multiply_8/Mul:z:03tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_8/Sum?
IdentityIdentity!tf.math.reduce_sum_8/Sum:output:0%^embedding_24/StatefulPartitionedCall%^embedding_25/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_24/StatefulPartitionedCall$embedding_24/StatefulPartitionedCall2L
$embedding_25/StatefulPartitionedCall$embedding_25/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_model_8_layer_call_fn_300118756
input_9
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2*
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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187452
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
_user_specified_name	input_9:

_output_shapes
: 
?	
?
K__inference_embedding_28_layer_call_and_return_conditional_losses_300120901

inputs
embedding_lookup_300120895
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_300120895Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/300120895*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/300120895*,
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
??
?
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300120056

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_14_greaterequal_y3
/model_8_tf_math_greater_equal_12_greaterequal_y3
/model_8_embedding_26_embedding_lookup_3001199063
/model_8_embedding_24_embedding_lookup_3001199123
/model_8_embedding_25_embedding_lookup_3001199203
/model_9_tf_math_greater_equal_13_greaterequal_y3
/model_9_embedding_29_embedding_lookup_3001199443
/model_9_embedding_27_embedding_lookup_3001199503
/model_9_embedding_28_embedding_lookup_300119958/
+tf_clip_by_value_14_clip_by_value_minimum_y'
#tf_clip_by_value_14_clip_by_value_y+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_39_matmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource+
'dense_40_matmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'dense_41_matmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource+
'dense_42_matmul_readvariableop_resource,
(dense_42_biasadd_readvariableop_resource+
'dense_43_matmul_readvariableop_resource,
(dense_43_biasadd_readvariableop_resource?
;normalize_4_normalization_4_reshape_readvariableop_resourceA
=normalize_4_normalization_4_reshape_1_readvariableop_resource+
'dense_44_matmul_readvariableop_resource,
(dense_44_biasadd_readvariableop_resource
identity??dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?dense_39/MatMul/ReadVariableOp?dense_40/BiasAdd/ReadVariableOp?dense_40/MatMul/ReadVariableOp?dense_41/BiasAdd/ReadVariableOp?dense_41/MatMul/ReadVariableOp?dense_42/BiasAdd/ReadVariableOp?dense_42/MatMul/ReadVariableOp?dense_43/BiasAdd/ReadVariableOp?dense_43/MatMul/ReadVariableOp?dense_44/BiasAdd/ReadVariableOp?dense_44/MatMul/ReadVariableOp?%model_8/embedding_24/embedding_lookup?%model_8/embedding_25/embedding_lookup?%model_8/embedding_26/embedding_lookup?%model_9/embedding_27/embedding_lookup?%model_9/embedding_28/embedding_lookup?%model_9/embedding_29/embedding_lookup?2normalize_4/normalization_4/Reshape/ReadVariableOp?4normalize_4/normalization_4/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_14/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_14_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_14/GreaterEqual?
model_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_8/flatten_8/Const?
model_8/flatten_8/ReshapeReshape
inputs_0_0 model_8/flatten_8/Const:output:0*
T0*'
_output_shapes
:?????????2
model_8/flatten_8/Reshape?
3model_8/tf.clip_by_value_12/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_8/tf.clip_by_value_12/clip_by_value/Minimum/y?
1model_8/tf.clip_by_value_12/clip_by_value/MinimumMinimum"model_8/flatten_8/Reshape:output:0<model_8/tf.clip_by_value_12/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_8/tf.clip_by_value_12/clip_by_value/Minimum?
+model_8/tf.clip_by_value_12/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_8/tf.clip_by_value_12/clip_by_value/y?
)model_8/tf.clip_by_value_12/clip_by_valueMaximum5model_8/tf.clip_by_value_12/clip_by_value/Minimum:z:04model_8/tf.clip_by_value_12/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_8/tf.clip_by_value_12/clip_by_value?
+model_8/tf.compat.v1.floor_div_8/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_8/tf.compat.v1.floor_div_8/FloorDiv/y?
)model_8/tf.compat.v1.floor_div_8/FloorDivFloorDiv-model_8/tf.clip_by_value_12/clip_by_value:z:04model_8/tf.compat.v1.floor_div_8/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_8/tf.compat.v1.floor_div_8/FloorDiv?
-model_8/tf.math.greater_equal_12/GreaterEqualGreaterEqual"model_8/flatten_8/Reshape:output:0/model_8_tf_math_greater_equal_12_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_8/tf.math.greater_equal_12/GreaterEqual?
%model_8/tf.math.floormod_8/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_8/tf.math.floormod_8/FloorMod/y?
#model_8/tf.math.floormod_8/FloorModFloorMod-model_8/tf.clip_by_value_12/clip_by_value:z:0.model_8/tf.math.floormod_8/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_8/tf.math.floormod_8/FloorMod?
model_8/embedding_26/CastCast-model_8/tf.clip_by_value_12/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_8/embedding_26/Cast?
%model_8/embedding_26/embedding_lookupResourceGather/model_8_embedding_26_embedding_lookup_300119906model_8/embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_8/embedding_26/embedding_lookup/300119906*,
_output_shapes
:??????????*
dtype02'
%model_8/embedding_26/embedding_lookup?
.model_8/embedding_26/embedding_lookup/IdentityIdentity.model_8/embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_8/embedding_26/embedding_lookup/300119906*,
_output_shapes
:??????????20
.model_8/embedding_26/embedding_lookup/Identity?
0model_8/embedding_26/embedding_lookup/Identity_1Identity7model_8/embedding_26/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_8/embedding_26/embedding_lookup/Identity_1?
model_8/embedding_24/CastCast-model_8/tf.compat.v1.floor_div_8/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_8/embedding_24/Cast?
%model_8/embedding_24/embedding_lookupResourceGather/model_8_embedding_24_embedding_lookup_300119912model_8/embedding_24/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_8/embedding_24/embedding_lookup/300119912*,
_output_shapes
:??????????*
dtype02'
%model_8/embedding_24/embedding_lookup?
.model_8/embedding_24/embedding_lookup/IdentityIdentity.model_8/embedding_24/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_8/embedding_24/embedding_lookup/300119912*,
_output_shapes
:??????????20
.model_8/embedding_24/embedding_lookup/Identity?
0model_8/embedding_24/embedding_lookup/Identity_1Identity7model_8/embedding_24/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_8/embedding_24/embedding_lookup/Identity_1?
model_8/tf.cast_12/CastCast1model_8/tf.math.greater_equal_12/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_8/tf.cast_12/Cast?
%model_8/tf.__operators__.add_24/AddV2AddV29model_8/embedding_26/embedding_lookup/Identity_1:output:09model_8/embedding_24/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_8/tf.__operators__.add_24/AddV2?
model_8/embedding_25/CastCast'model_8/tf.math.floormod_8/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_8/embedding_25/Cast?
%model_8/embedding_25/embedding_lookupResourceGather/model_8_embedding_25_embedding_lookup_300119920model_8/embedding_25/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_8/embedding_25/embedding_lookup/300119920*,
_output_shapes
:??????????*
dtype02'
%model_8/embedding_25/embedding_lookup?
.model_8/embedding_25/embedding_lookup/IdentityIdentity.model_8/embedding_25/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_8/embedding_25/embedding_lookup/300119920*,
_output_shapes
:??????????20
.model_8/embedding_25/embedding_lookup/Identity?
0model_8/embedding_25/embedding_lookup/Identity_1Identity7model_8/embedding_25/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_8/embedding_25/embedding_lookup/Identity_1?
%model_8/tf.__operators__.add_25/AddV2AddV2)model_8/tf.__operators__.add_24/AddV2:z:09model_8/embedding_25/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_8/tf.__operators__.add_25/AddV2?
'model_8/tf.expand_dims_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_8/tf.expand_dims_8/ExpandDims/dim?
#model_8/tf.expand_dims_8/ExpandDims
ExpandDimsmodel_8/tf.cast_12/Cast:y:00model_8/tf.expand_dims_8/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_8/tf.expand_dims_8/ExpandDims?
model_8/tf.math.multiply_8/MulMul)model_8/tf.__operators__.add_25/AddV2:z:0,model_8/tf.expand_dims_8/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_8/tf.math.multiply_8/Mul?
2model_8/tf.math.reduce_sum_8/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_8/tf.math.reduce_sum_8/Sum/reduction_indices?
 model_8/tf.math.reduce_sum_8/SumSum"model_8/tf.math.multiply_8/Mul:z:0;model_8/tf.math.reduce_sum_8/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_8/tf.math.reduce_sum_8/Sum?
model_9/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_9/flatten_9/Const?
model_9/flatten_9/ReshapeReshape
inputs_0_1 model_9/flatten_9/Const:output:0*
T0*'
_output_shapes
:?????????2
model_9/flatten_9/Reshape?
3model_9/tf.clip_by_value_13/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_9/tf.clip_by_value_13/clip_by_value/Minimum/y?
1model_9/tf.clip_by_value_13/clip_by_value/MinimumMinimum"model_9/flatten_9/Reshape:output:0<model_9/tf.clip_by_value_13/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_9/tf.clip_by_value_13/clip_by_value/Minimum?
+model_9/tf.clip_by_value_13/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_9/tf.clip_by_value_13/clip_by_value/y?
)model_9/tf.clip_by_value_13/clip_by_valueMaximum5model_9/tf.clip_by_value_13/clip_by_value/Minimum:z:04model_9/tf.clip_by_value_13/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_9/tf.clip_by_value_13/clip_by_value?
+model_9/tf.compat.v1.floor_div_9/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_9/tf.compat.v1.floor_div_9/FloorDiv/y?
)model_9/tf.compat.v1.floor_div_9/FloorDivFloorDiv-model_9/tf.clip_by_value_13/clip_by_value:z:04model_9/tf.compat.v1.floor_div_9/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_9/tf.compat.v1.floor_div_9/FloorDiv?
-model_9/tf.math.greater_equal_13/GreaterEqualGreaterEqual"model_9/flatten_9/Reshape:output:0/model_9_tf_math_greater_equal_13_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_9/tf.math.greater_equal_13/GreaterEqual?
%model_9/tf.math.floormod_9/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_9/tf.math.floormod_9/FloorMod/y?
#model_9/tf.math.floormod_9/FloorModFloorMod-model_9/tf.clip_by_value_13/clip_by_value:z:0.model_9/tf.math.floormod_9/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_9/tf.math.floormod_9/FloorMod?
model_9/embedding_29/CastCast-model_9/tf.clip_by_value_13/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_9/embedding_29/Cast?
%model_9/embedding_29/embedding_lookupResourceGather/model_9_embedding_29_embedding_lookup_300119944model_9/embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_9/embedding_29/embedding_lookup/300119944*,
_output_shapes
:??????????*
dtype02'
%model_9/embedding_29/embedding_lookup?
.model_9/embedding_29/embedding_lookup/IdentityIdentity.model_9/embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_9/embedding_29/embedding_lookup/300119944*,
_output_shapes
:??????????20
.model_9/embedding_29/embedding_lookup/Identity?
0model_9/embedding_29/embedding_lookup/Identity_1Identity7model_9/embedding_29/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_9/embedding_29/embedding_lookup/Identity_1?
model_9/embedding_27/CastCast-model_9/tf.compat.v1.floor_div_9/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_9/embedding_27/Cast?
%model_9/embedding_27/embedding_lookupResourceGather/model_9_embedding_27_embedding_lookup_300119950model_9/embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_9/embedding_27/embedding_lookup/300119950*,
_output_shapes
:??????????*
dtype02'
%model_9/embedding_27/embedding_lookup?
.model_9/embedding_27/embedding_lookup/IdentityIdentity.model_9/embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_9/embedding_27/embedding_lookup/300119950*,
_output_shapes
:??????????20
.model_9/embedding_27/embedding_lookup/Identity?
0model_9/embedding_27/embedding_lookup/Identity_1Identity7model_9/embedding_27/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_9/embedding_27/embedding_lookup/Identity_1?
model_9/tf.cast_13/CastCast1model_9/tf.math.greater_equal_13/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_9/tf.cast_13/Cast?
%model_9/tf.__operators__.add_26/AddV2AddV29model_9/embedding_29/embedding_lookup/Identity_1:output:09model_9/embedding_27/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_9/tf.__operators__.add_26/AddV2?
model_9/embedding_28/CastCast'model_9/tf.math.floormod_9/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_9/embedding_28/Cast?
%model_9/embedding_28/embedding_lookupResourceGather/model_9_embedding_28_embedding_lookup_300119958model_9/embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_9/embedding_28/embedding_lookup/300119958*,
_output_shapes
:??????????*
dtype02'
%model_9/embedding_28/embedding_lookup?
.model_9/embedding_28/embedding_lookup/IdentityIdentity.model_9/embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_9/embedding_28/embedding_lookup/300119958*,
_output_shapes
:??????????20
.model_9/embedding_28/embedding_lookup/Identity?
0model_9/embedding_28/embedding_lookup/Identity_1Identity7model_9/embedding_28/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_9/embedding_28/embedding_lookup/Identity_1?
%model_9/tf.__operators__.add_27/AddV2AddV2)model_9/tf.__operators__.add_26/AddV2:z:09model_9/embedding_28/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_9/tf.__operators__.add_27/AddV2?
'model_9/tf.expand_dims_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_9/tf.expand_dims_9/ExpandDims/dim?
#model_9/tf.expand_dims_9/ExpandDims
ExpandDimsmodel_9/tf.cast_13/Cast:y:00model_9/tf.expand_dims_9/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_9/tf.expand_dims_9/ExpandDims?
model_9/tf.math.multiply_9/MulMul)model_9/tf.__operators__.add_27/AddV2:z:0,model_9/tf.expand_dims_9/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_9/tf.math.multiply_9/Mul?
2model_9/tf.math.reduce_sum_9/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_9/tf.math.reduce_sum_9/Sum/reduction_indices?
 model_9/tf.math.reduce_sum_9/SumSum"model_9/tf.math.multiply_9/Mul:z:0;model_9/tf.math.reduce_sum_9/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_9/tf.math.reduce_sum_9/Sum?
)tf.clip_by_value_14/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_14_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_14/clip_by_value/Minimum?
!tf.clip_by_value_14/clip_by_valueMaximum-tf.clip_by_value_14/clip_by_value/Minimum:z:0#tf_clip_by_value_14_clip_by_value_y*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_14/clip_by_value?
tf.cast_14/CastCast)tf.math.greater_equal_14/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_14/Castv
tf.concat_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_12/concat/axis?
tf.concat_12/concatConcatV2)model_8/tf.math.reduce_sum_8/Sum:output:0)model_9/tf.math.reduce_sum_9/Sum:output:0!tf.concat_12/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_12/concat
tf.concat_13/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_13/concat/axis?
tf.concat_13/concatConcatV2%tf.clip_by_value_14/clip_by_value:z:0tf.cast_14/Cast:y:0!tf.concat_13/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_13/concat?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMultf.concat_12/concat:output:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_36/BiasAddt
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_36/Relu?
dense_39/MatMul/ReadVariableOpReadVariableOp'dense_39_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_39/MatMul/ReadVariableOp?
dense_39/MatMulMatMultf.concat_13/concat:output:0&dense_39/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_39/MatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_39/BiasAdd?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Relu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_37/BiasAddt
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_37/Relu?
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_38/MatMul/ReadVariableOp?
dense_38/MatMulMatMuldense_37/Relu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_38/MatMul?
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_38/BiasAdd/ReadVariableOp?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_38/BiasAddt
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_38/Relu?
dense_40/MatMul/ReadVariableOpReadVariableOp'dense_40_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_40/MatMul/ReadVariableOp?
dense_40/MatMulMatMuldense_39/BiasAdd:output:0&dense_40/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_40/MatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_40/BiasAdd
tf.concat_14/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_14/concat/axis?
tf.concat_14/concatConcatV2dense_38/Relu:activations:0dense_40/BiasAdd:output:0!tf.concat_14/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_14/concat?
dense_41/MatMul/ReadVariableOpReadVariableOp'dense_41_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_41/MatMul/ReadVariableOp?
dense_41/MatMulMatMultf.concat_14/concat:output:0&dense_41/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/MatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_41/BiasAdd~
tf.nn.relu_12/ReluReludense_41/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_12/Relu?
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_42/MatMul/ReadVariableOp?
dense_42/MatMulMatMul tf.nn.relu_12/Relu:activations:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/MatMul?
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_42/BiasAdd/ReadVariableOp?
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_42/BiasAdd?
tf.__operators__.add_28/AddV2AddV2dense_42/BiasAdd:output:0 tf.nn.relu_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_28/AddV2?
tf.nn.relu_13/ReluRelu!tf.__operators__.add_28/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_13/Relu?
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_43/MatMul/ReadVariableOp?
dense_43/MatMulMatMul tf.nn.relu_13/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/MatMul?
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_43/BiasAdd/ReadVariableOp?
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_43/BiasAdd?
tf.__operators__.add_29/AddV2AddV2dense_43/BiasAdd:output:0 tf.nn.relu_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_29/AddV2?
tf.nn.relu_14/ReluRelu!tf.__operators__.add_29/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_14/Relu?
2normalize_4/normalization_4/Reshape/ReadVariableOpReadVariableOp;normalize_4_normalization_4_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_4/normalization_4/Reshape/ReadVariableOp?
)normalize_4/normalization_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_4/normalization_4/Reshape/shape?
#normalize_4/normalization_4/ReshapeReshape:normalize_4/normalization_4/Reshape/ReadVariableOp:value:02normalize_4/normalization_4/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_4/normalization_4/Reshape?
4normalize_4/normalization_4/Reshape_1/ReadVariableOpReadVariableOp=normalize_4_normalization_4_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_4/normalization_4/Reshape_1/ReadVariableOp?
+normalize_4/normalization_4/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_4/normalization_4/Reshape_1/shape?
%normalize_4/normalization_4/Reshape_1Reshape<normalize_4/normalization_4/Reshape_1/ReadVariableOp:value:04normalize_4/normalization_4/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_4/normalization_4/Reshape_1?
normalize_4/normalization_4/subSub tf.nn.relu_14/Relu:activations:0,normalize_4/normalization_4/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_4/normalization_4/sub?
 normalize_4/normalization_4/SqrtSqrt.normalize_4/normalization_4/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_4/normalization_4/Sqrt?
%normalize_4/normalization_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_4/normalization_4/Maximum/y?
#normalize_4/normalization_4/MaximumMaximum$normalize_4/normalization_4/Sqrt:y:0.normalize_4/normalization_4/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_4/normalization_4/Maximum?
#normalize_4/normalization_4/truedivRealDiv#normalize_4/normalization_4/sub:z:0'normalize_4/normalization_4/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_4/normalization_4/truediv?
dense_44/MatMul/ReadVariableOpReadVariableOp'dense_44_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_44/MatMul/ReadVariableOp?
dense_44/MatMulMatMul'normalize_4/normalization_4/truediv:z:0&dense_44/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_44/MatMul?
dense_44/BiasAdd/ReadVariableOpReadVariableOp(dense_44_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_44/BiasAdd/ReadVariableOp?
dense_44/BiasAddBiasAdddense_44/MatMul:product:0'dense_44/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_44/BiasAdd?
IdentityIdentitydense_44/BiasAdd:output:0 ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp^dense_39/MatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp^dense_40/MatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp^dense_41/MatMul/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp ^dense_44/BiasAdd/ReadVariableOp^dense_44/MatMul/ReadVariableOp&^model_8/embedding_24/embedding_lookup&^model_8/embedding_25/embedding_lookup&^model_8/embedding_26/embedding_lookup&^model_9/embedding_27/embedding_lookup&^model_9/embedding_28/embedding_lookup&^model_9/embedding_29/embedding_lookup3^normalize_4/normalization_4/Reshape/ReadVariableOp5^normalize_4/normalization_4/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????: : :::: :::: : ::::::::::::::::::::2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2@
dense_39/MatMul/ReadVariableOpdense_39/MatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2@
dense_40/MatMul/ReadVariableOpdense_40/MatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2@
dense_41/MatMul/ReadVariableOpdense_41/MatMul/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp2B
dense_44/BiasAdd/ReadVariableOpdense_44/BiasAdd/ReadVariableOp2@
dense_44/MatMul/ReadVariableOpdense_44/MatMul/ReadVariableOp2N
%model_8/embedding_24/embedding_lookup%model_8/embedding_24/embedding_lookup2N
%model_8/embedding_25/embedding_lookup%model_8/embedding_25/embedding_lookup2N
%model_8/embedding_26/embedding_lookup%model_8/embedding_26/embedding_lookup2N
%model_9/embedding_27/embedding_lookup%model_9/embedding_27/embedding_lookup2N
%model_9/embedding_28/embedding_lookup%model_9/embedding_28/embedding_lookup2N
%model_9/embedding_29/embedding_lookup%model_9/embedding_29/embedding_lookup2h
2normalize_4/normalization_4/Reshape/ReadVariableOp2normalize_4/normalization_4/Reshape/ReadVariableOp2l
4normalize_4/normalization_4/Reshape_1/ReadVariableOp4normalize_4/normalization_4/Reshape_1/ReadVariableOp:S O
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
??
?#
"__inference__traced_save_300121184
file_prefix.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop.
*savev2_dense_42_kernel_read_readvariableop,
(savev2_dense_42_bias_read_readvariableop.
*savev2_dense_43_kernel_read_readvariableop,
(savev2_dense_43_bias_read_readvariableop.
*savev2_dense_44_kernel_read_readvariableop,
(savev2_dense_44_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_embedding_26_embeddings_read_readvariableop6
2savev2_embedding_24_embeddings_read_readvariableop6
2savev2_embedding_25_embeddings_read_readvariableop6
2savev2_embedding_29_embeddings_read_readvariableop6
2savev2_embedding_27_embeddings_read_readvariableop6
2savev2_embedding_28_embeddings_read_readvariableop?
;savev2_normalize_4_normalization_4_mean_read_readvariableopC
?savev2_normalize_4_normalization_4_variance_read_readvariableop@
<savev2_normalize_4_normalization_4_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableop5
1savev2_adam_dense_38_kernel_m_read_readvariableop3
/savev2_adam_dense_38_bias_m_read_readvariableop5
1savev2_adam_dense_40_kernel_m_read_readvariableop3
/savev2_adam_dense_40_bias_m_read_readvariableop5
1savev2_adam_dense_41_kernel_m_read_readvariableop3
/savev2_adam_dense_41_bias_m_read_readvariableop5
1savev2_adam_dense_42_kernel_m_read_readvariableop3
/savev2_adam_dense_42_bias_m_read_readvariableop5
1savev2_adam_dense_43_kernel_m_read_readvariableop3
/savev2_adam_dense_43_bias_m_read_readvariableop5
1savev2_adam_dense_44_kernel_m_read_readvariableop3
/savev2_adam_dense_44_bias_m_read_readvariableop=
9savev2_adam_embedding_26_embeddings_m_read_readvariableop=
9savev2_adam_embedding_24_embeddings_m_read_readvariableop=
9savev2_adam_embedding_25_embeddings_m_read_readvariableop=
9savev2_adam_embedding_29_embeddings_m_read_readvariableop=
9savev2_adam_embedding_27_embeddings_m_read_readvariableop=
9savev2_adam_embedding_28_embeddings_m_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableop5
1savev2_adam_dense_38_kernel_v_read_readvariableop3
/savev2_adam_dense_38_bias_v_read_readvariableop5
1savev2_adam_dense_40_kernel_v_read_readvariableop3
/savev2_adam_dense_40_bias_v_read_readvariableop5
1savev2_adam_dense_41_kernel_v_read_readvariableop3
/savev2_adam_dense_41_bias_v_read_readvariableop5
1savev2_adam_dense_42_kernel_v_read_readvariableop3
/savev2_adam_dense_42_bias_v_read_readvariableop5
1savev2_adam_dense_43_kernel_v_read_readvariableop3
/savev2_adam_dense_43_bias_v_read_readvariableop5
1savev2_adam_dense_44_kernel_v_read_readvariableop3
/savev2_adam_dense_44_bias_v_read_readvariableop=
9savev2_adam_embedding_26_embeddings_v_read_readvariableop=
9savev2_adam_embedding_24_embeddings_v_read_readvariableop=
9savev2_adam_embedding_25_embeddings_v_read_readvariableop=
9savev2_adam_embedding_29_embeddings_v_read_readvariableop=
9savev2_adam_embedding_27_embeddings_v_read_readvariableop=
9savev2_adam_embedding_28_embeddings_v_read_readvariableop
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
ShardedFilename?,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?+
value?+B?+SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop*savev2_dense_42_kernel_read_readvariableop(savev2_dense_42_bias_read_readvariableop*savev2_dense_43_kernel_read_readvariableop(savev2_dense_43_bias_read_readvariableop*savev2_dense_44_kernel_read_readvariableop(savev2_dense_44_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_embedding_26_embeddings_read_readvariableop2savev2_embedding_24_embeddings_read_readvariableop2savev2_embedding_25_embeddings_read_readvariableop2savev2_embedding_29_embeddings_read_readvariableop2savev2_embedding_27_embeddings_read_readvariableop2savev2_embedding_28_embeddings_read_readvariableop;savev2_normalize_4_normalization_4_mean_read_readvariableop?savev2_normalize_4_normalization_4_variance_read_readvariableop<savev2_normalize_4_normalization_4_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableop1savev2_adam_dense_38_kernel_m_read_readvariableop/savev2_adam_dense_38_bias_m_read_readvariableop1savev2_adam_dense_40_kernel_m_read_readvariableop/savev2_adam_dense_40_bias_m_read_readvariableop1savev2_adam_dense_41_kernel_m_read_readvariableop/savev2_adam_dense_41_bias_m_read_readvariableop1savev2_adam_dense_42_kernel_m_read_readvariableop/savev2_adam_dense_42_bias_m_read_readvariableop1savev2_adam_dense_43_kernel_m_read_readvariableop/savev2_adam_dense_43_bias_m_read_readvariableop1savev2_adam_dense_44_kernel_m_read_readvariableop/savev2_adam_dense_44_bias_m_read_readvariableop9savev2_adam_embedding_26_embeddings_m_read_readvariableop9savev2_adam_embedding_24_embeddings_m_read_readvariableop9savev2_adam_embedding_25_embeddings_m_read_readvariableop9savev2_adam_embedding_29_embeddings_m_read_readvariableop9savev2_adam_embedding_27_embeddings_m_read_readvariableop9savev2_adam_embedding_28_embeddings_m_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableop1savev2_adam_dense_38_kernel_v_read_readvariableop/savev2_adam_dense_38_bias_v_read_readvariableop1savev2_adam_dense_40_kernel_v_read_readvariableop/savev2_adam_dense_40_bias_v_read_readvariableop1savev2_adam_dense_41_kernel_v_read_readvariableop/savev2_adam_dense_41_bias_v_read_readvariableop1savev2_adam_dense_42_kernel_v_read_readvariableop/savev2_adam_dense_42_bias_v_read_readvariableop1savev2_adam_dense_43_kernel_v_read_readvariableop/savev2_adam_dense_43_bias_v_read_readvariableop1savev2_adam_dense_44_kernel_v_read_readvariableop/savev2_adam_dense_44_bias_v_read_readvariableop9savev2_adam_embedding_26_embeddings_v_read_readvariableop9savev2_adam_embedding_24_embeddings_v_read_readvariableop9savev2_adam_embedding_25_embeddings_v_read_readvariableop9savev2_adam_embedding_29_embeddings_v_read_readvariableop9savev2_adam_embedding_27_embeddings_v_read_readvariableop9savev2_adam_embedding_28_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S		2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?:: : : : : :	4?:	?:	?:	4?:	?:	?:?:?: : : :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?:
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?: 2(
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
:	?:!
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
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?: 4

_output_shapes
::%5!

_output_shapes
:	4?:%6!

_output_shapes
:	?:%7!

_output_shapes
:	?:%8!

_output_shapes
:	4?:%9!

_output_shapes
:	?:%:!

_output_shapes
:	?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:%?!

_output_shapes
:	?:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:&C"
 
_output_shapes
:
??:!D

_output_shapes	
:?:&E"
 
_output_shapes
:
??:!F

_output_shapes	
:?:&G"
 
_output_shapes
:
??:!H

_output_shapes	
:?:&I"
 
_output_shapes
:
??:!J

_output_shapes	
:?:%K!

_output_shapes
:	?: L

_output_shapes
::%M!

_output_shapes
:	4?:%N!

_output_shapes
:	?:%O!

_output_shapes
:	?:%P!

_output_shapes
:	4?:%Q!

_output_shapes
:	?:%R!

_output_shapes
:	?:S

_output_shapes
: 
?	
?
G__inference_dense_43_layer_call_and_return_conditional_losses_300119314

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
?
,__inference_dense_41_layer_call_fn_300120701

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
GPU2 *0J 8? *P
fKRI
G__inference_dense_41_layer_call_and_return_conditional_losses_3001192592
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
?
?
+__inference_model_8_layer_call_fn_300120461

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
GPU2 *0J 8? *O
fJRH
F__inference_model_8_layer_call_and_return_conditional_losses_3001187452
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
?	
?
G__inference_dense_43_layer_call_and_return_conditional_losses_300120730

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
G__inference_dense_41_layer_call_and_return_conditional_losses_300119259

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
 
_user_specified_nameinputs"?L
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
serving_default_cards1:0?????????<
dense_440
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
	optimizer
loss
regularization_losses
	variables
trainable_variables
 	keras_api
!
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_network??{"class_name": "CustomModel", "name": "custom_model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_12", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_12", "inbound_nodes": [["flatten_8", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_8", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_26", "inbound_nodes": [[["tf.clip_by_value_12", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_24", "inbound_nodes": [[["tf.compat.v1.floor_div_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_8", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_12", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_12", "inbound_nodes": [["flatten_8", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_24", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_24", "inbound_nodes": [["embedding_26", 0, 0, {"y": ["embedding_24", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_25", "inbound_nodes": [[["tf.math.floormod_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_12", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_12", "inbound_nodes": [["tf.math.greater_equal_12", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_25", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_25", "inbound_nodes": [["tf.__operators__.add_24", 0, 0, {"y": ["embedding_25", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_8", "inbound_nodes": [["tf.cast_12", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["tf.__operators__.add_25", 0, 0, {"y": ["tf.expand_dims_8", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"axis": 1}]]}], "input_layers": [["input_9", 0, 0]], "output_layers": [["tf.math.reduce_sum_8", 0, 0]]}, "name": "model_8", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_13", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_13", "inbound_nodes": [["flatten_9", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_9", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_29", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_29", "inbound_nodes": [[["tf.clip_by_value_13", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_27", "inbound_nodes": [[["tf.compat.v1.floor_div_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_9", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_13", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_13", "inbound_nodes": [["flatten_9", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_26", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_26", "inbound_nodes": [["embedding_29", 0, 0, {"y": ["embedding_27", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_28", "inbound_nodes": [[["tf.math.floormod_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_13", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_13", "inbound_nodes": [["tf.math.greater_equal_13", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_27", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_27", "inbound_nodes": [["tf.__operators__.add_26", 0, 0, {"y": ["embedding_28", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_9", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_9", "inbound_nodes": [["tf.cast_13", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["tf.__operators__.add_27", 0, 0, {"y": ["tf.expand_dims_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_9", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"axis": 1}]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["tf.math.reduce_sum_9", 0, 0]]}, "name": "model_9", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_14", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_14", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_12", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_12", "inbound_nodes": [[["model_8", 1, 0, {"axis": 1}], ["model_9", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_14", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_14", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_14", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_14", "inbound_nodes": [["tf.math.greater_equal_14", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_36", "inbound_nodes": [[["tf.concat_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_13", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_13", "inbound_nodes": [[["tf.clip_by_value_14", 0, 0, {"axis": -1}], ["tf.cast_14", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_37", "inbound_nodes": [[["dense_36", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["tf.concat_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_38", "inbound_nodes": [[["dense_37", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_14", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_14", "inbound_nodes": [[["dense_38", 0, 0, {"axis": -1}], ["dense_40", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_41", "inbound_nodes": [[["tf.concat_14", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_12", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_12", "inbound_nodes": [["dense_41", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_42", "inbound_nodes": [[["tf.nn.relu_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_28", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_28", "inbound_nodes": [["dense_42", 0, 0, {"y": ["tf.nn.relu_12", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_13", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_13", "inbound_nodes": [["tf.__operators__.add_28", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_43", "inbound_nodes": [[["tf.nn.relu_13", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_29", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_29", "inbound_nodes": [["dense_43", 0, 0, {"y": ["tf.nn.relu_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_14", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_14", "inbound_nodes": [["tf.__operators__.add_29", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_4", "trainable": true, "dtype": "float32"}, "name": "normalize_4", "inbound_nodes": [[["tf.nn.relu_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_44", "inbound_nodes": [[["normalize_4", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_44", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 12]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_12", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_12", "inbound_nodes": [["flatten_8", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_8", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_26", "inbound_nodes": [[["tf.clip_by_value_12", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_24", "inbound_nodes": [[["tf.compat.v1.floor_div_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_8", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_12", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_12", "inbound_nodes": [["flatten_8", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_24", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_24", "inbound_nodes": [["embedding_26", 0, 0, {"y": ["embedding_24", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_25", "inbound_nodes": [[["tf.math.floormod_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_12", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_12", "inbound_nodes": [["tf.math.greater_equal_12", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_25", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_25", "inbound_nodes": [["tf.__operators__.add_24", 0, 0, {"y": ["embedding_25", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_8", "inbound_nodes": [["tf.cast_12", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["tf.__operators__.add_25", 0, 0, {"y": ["tf.expand_dims_8", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"axis": 1}]]}], "input_layers": [["input_9", 0, 0]], "output_layers": [["tf.math.reduce_sum_8", 0, 0]]}, "name": "model_8", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_13", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_13", "inbound_nodes": [["flatten_9", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_9", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_29", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_29", "inbound_nodes": [[["tf.clip_by_value_13", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_27", "inbound_nodes": [[["tf.compat.v1.floor_div_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_9", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_13", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_13", "inbound_nodes": [["flatten_9", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_26", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_26", "inbound_nodes": [["embedding_29", 0, 0, {"y": ["embedding_27", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_28", "inbound_nodes": [[["tf.math.floormod_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_13", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_13", "inbound_nodes": [["tf.math.greater_equal_13", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_27", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_27", "inbound_nodes": [["tf.__operators__.add_26", 0, 0, {"y": ["embedding_28", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_9", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_9", "inbound_nodes": [["tf.cast_13", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["tf.__operators__.add_27", 0, 0, {"y": ["tf.expand_dims_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_9", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"axis": 1}]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["tf.math.reduce_sum_9", 0, 0]]}, "name": "model_9", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_14", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_14", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_12", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_12", "inbound_nodes": [[["model_8", 1, 0, {"axis": 1}], ["model_9", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_14", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_14", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_14", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_14", "inbound_nodes": [["tf.math.greater_equal_14", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_36", "inbound_nodes": [[["tf.concat_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_13", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_13", "inbound_nodes": [[["tf.clip_by_value_14", 0, 0, {"axis": -1}], ["tf.cast_14", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_37", "inbound_nodes": [[["dense_36", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["tf.concat_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_38", "inbound_nodes": [[["dense_37", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_14", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_14", "inbound_nodes": [[["dense_38", 0, 0, {"axis": -1}], ["dense_40", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_41", "inbound_nodes": [[["tf.concat_14", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_12", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_12", "inbound_nodes": [["dense_41", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_42", "inbound_nodes": [[["tf.nn.relu_12", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_28", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_28", "inbound_nodes": [["dense_42", 0, 0, {"y": ["tf.nn.relu_12", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_13", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_13", "inbound_nodes": [["tf.__operators__.add_28", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_43", "inbound_nodes": [[["tf.nn.relu_13", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_29", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_29", "inbound_nodes": [["dense_43", 0, 0, {"y": ["tf.nn.relu_13", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_14", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_14", "inbound_nodes": [["tf.__operators__.add_29", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_4", "trainable": true, "dtype": "float32"}, "name": "normalize_4", "inbound_nodes": [[["tf.nn.relu_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_44", "inbound_nodes": [[["normalize_4", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_44", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?Q
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_12", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_12", "inbound_nodes": [["flatten_8", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_8", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_26", "inbound_nodes": [[["tf.clip_by_value_12", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_24", "inbound_nodes": [[["tf.compat.v1.floor_div_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_8", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_12", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_12", "inbound_nodes": [["flatten_8", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_24", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_24", "inbound_nodes": [["embedding_26", 0, 0, {"y": ["embedding_24", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_25", "inbound_nodes": [[["tf.math.floormod_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_12", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_12", "inbound_nodes": [["tf.math.greater_equal_12", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_25", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_25", "inbound_nodes": [["tf.__operators__.add_24", 0, 0, {"y": ["embedding_25", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_8", "inbound_nodes": [["tf.cast_12", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["tf.__operators__.add_25", 0, 0, {"y": ["tf.expand_dims_8", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"axis": 1}]]}], "input_layers": [["input_9", 0, 0]], "output_layers": [["tf.math.reduce_sum_8", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}, "name": "input_9", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["input_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_12", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_12", "inbound_nodes": [["flatten_8", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_8", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_26", "inbound_nodes": [[["tf.clip_by_value_12", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_24", "inbound_nodes": [[["tf.compat.v1.floor_div_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_8", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_8", "inbound_nodes": [["tf.clip_by_value_12", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_12", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_12", "inbound_nodes": [["flatten_8", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_24", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_24", "inbound_nodes": [["embedding_26", 0, 0, {"y": ["embedding_24", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_25", "inbound_nodes": [[["tf.math.floormod_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_12", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_12", "inbound_nodes": [["tf.math.greater_equal_12", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_25", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_25", "inbound_nodes": [["tf.__operators__.add_24", 0, 0, {"y": ["embedding_25", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_8", "inbound_nodes": [["tf.cast_12", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_8", "inbound_nodes": [["tf.__operators__.add_25", 0, 0, {"y": ["tf.expand_dims_8", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_8", "inbound_nodes": [["tf.math.multiply_8", 0, 0, {"axis": 1}]]}], "input_layers": [["input_9", 0, 0]], "output_layers": [["tf.math.reduce_sum_8", 0, 0]]}}}
?Q
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_13", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_13", "inbound_nodes": [["flatten_9", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_9", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_29", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_29", "inbound_nodes": [[["tf.clip_by_value_13", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_27", "inbound_nodes": [[["tf.compat.v1.floor_div_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_9", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_13", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_13", "inbound_nodes": [["flatten_9", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_26", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_26", "inbound_nodes": [["embedding_29", 0, 0, {"y": ["embedding_27", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_28", "inbound_nodes": [[["tf.math.floormod_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_13", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_13", "inbound_nodes": [["tf.math.greater_equal_13", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_27", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_27", "inbound_nodes": [["tf.__operators__.add_26", 0, 0, {"y": ["embedding_28", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_9", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_9", "inbound_nodes": [["tf.cast_13", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["tf.__operators__.add_27", 0, 0, {"y": ["tf.expand_dims_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_9", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"axis": 1}]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["tf.math.reduce_sum_9", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}, "name": "input_10", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_9", "inbound_nodes": [[["input_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_13", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_13", "inbound_nodes": [["flatten_9", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_9", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_29", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_29", "inbound_nodes": [[["tf.clip_by_value_13", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_27", "inbound_nodes": [[["tf.compat.v1.floor_div_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_9", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_9", "inbound_nodes": [["tf.clip_by_value_13", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_13", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_13", "inbound_nodes": [["flatten_9", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_26", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_26", "inbound_nodes": [["embedding_29", 0, 0, {"y": ["embedding_27", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_28", "inbound_nodes": [[["tf.math.floormod_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_13", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_13", "inbound_nodes": [["tf.math.greater_equal_13", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_27", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_27", "inbound_nodes": [["tf.__operators__.add_26", 0, 0, {"y": ["embedding_28", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_9", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_9", "inbound_nodes": [["tf.cast_13", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_9", "inbound_nodes": [["tf.__operators__.add_27", 0, 0, {"y": ["tf.expand_dims_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_9", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_9", "inbound_nodes": [["tf.math.multiply_9", 0, 0, {"axis": 1}]]}], "input_layers": [["input_10", 0, 0]], "output_layers": [["tf.math.reduce_sum_9", 0, 0]]}}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_14", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_12", "trainable": true, "dtype": "float32", "function": "concat"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_14", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_14", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_13", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24]}}
?

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
k	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_14", "trainable": true, "dtype": "float32", "function": "concat"}}
?

lkernel
mbias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
r	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_12", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

skernel
tbias
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_28", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_13", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_29", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_14", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_4", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_44", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
regularization_losses
	variables
?metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_9", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_12", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_8", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_26", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_24", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_8", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_12", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_24", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_12", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_25", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_8", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_8", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_8", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
1regularization_losses
2	variables
?metrics
3trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_10", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_10"}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_9", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_13", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_9", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_29", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_27", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_9", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_13", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_26", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_28", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_13", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_27", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_9", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_9", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_9", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
Dregularization_losses
E	variables
?metrics
Ftrainable_variables
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
#:!
??2dense_36/kernel
:?2dense_36/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
Nregularization_losses
O	variables
?metrics
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_37/kernel
:?2dense_37/bias
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
Uregularization_losses
V	variables
?metrics
Wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_39/kernel
:?2dense_39/bias
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
[regularization_losses
\	variables
?metrics
]trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_38/kernel
:?2dense_38/bias
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
aregularization_losses
b	variables
?metrics
ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_40/kernel
:?2dense_40/bias
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
gregularization_losses
h	variables
?metrics
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_41/kernel
:?2dense_41/bias
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
nregularization_losses
o	variables
?metrics
ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_42/kernel
:?2dense_42/bias
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
uregularization_losses
v	variables
?metrics
wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_43/kernel
:?2dense_43/bias
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
}regularization_losses
~	variables
?metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_4", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_44/kernel
:2dense_44/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	4?2embedding_26/embeddings
*:(	?2embedding_24/embeddings
*:(	?2embedding_25/embeddings
*:(	4?2embedding_29/embeddings
*:(	?2embedding_27/embeddings
*:(	?2embedding_28/embeddings
-:+?2 normalize_4/normalization_4/mean
1:/?2$normalize_4/normalization_4/variance
):'	 2!normalize_4/normalization_4/count
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
?layers
?regularization_losses
?	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14"
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
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_36/kernel/m
!:?2Adam/dense_36/bias/m
(:&
??2Adam/dense_37/kernel/m
!:?2Adam/dense_37/bias/m
':%	?2Adam/dense_39/kernel/m
!:?2Adam/dense_39/bias/m
(:&
??2Adam/dense_38/kernel/m
!:?2Adam/dense_38/bias/m
(:&
??2Adam/dense_40/kernel/m
!:?2Adam/dense_40/bias/m
(:&
??2Adam/dense_41/kernel/m
!:?2Adam/dense_41/bias/m
(:&
??2Adam/dense_42/kernel/m
!:?2Adam/dense_42/bias/m
(:&
??2Adam/dense_43/kernel/m
!:?2Adam/dense_43/bias/m
':%	?2Adam/dense_44/kernel/m
 :2Adam/dense_44/bias/m
/:-	4?2Adam/embedding_26/embeddings/m
/:-	?2Adam/embedding_24/embeddings/m
/:-	?2Adam/embedding_25/embeddings/m
/:-	4?2Adam/embedding_29/embeddings/m
/:-	?2Adam/embedding_27/embeddings/m
/:-	?2Adam/embedding_28/embeddings/m
(:&
??2Adam/dense_36/kernel/v
!:?2Adam/dense_36/bias/v
(:&
??2Adam/dense_37/kernel/v
!:?2Adam/dense_37/bias/v
':%	?2Adam/dense_39/kernel/v
!:?2Adam/dense_39/bias/v
(:&
??2Adam/dense_38/kernel/v
!:?2Adam/dense_38/bias/v
(:&
??2Adam/dense_40/kernel/v
!:?2Adam/dense_40/bias/v
(:&
??2Adam/dense_41/kernel/v
!:?2Adam/dense_41/bias/v
(:&
??2Adam/dense_42/kernel/v
!:?2Adam/dense_42/bias/v
(:&
??2Adam/dense_43/kernel/v
!:?2Adam/dense_43/bias/v
':%	?2Adam/dense_44/kernel/v
 :2Adam/dense_44/bias/v
/:-	4?2Adam/embedding_26/embeddings/v
/:-	?2Adam/embedding_24/embeddings/v
/:-	?2Adam/embedding_25/embeddings/v
/:-	4?2Adam/embedding_29/embeddings/v
/:-	?2Adam/embedding_27/embeddings/v
/:-	?2Adam/embedding_28/embeddings/v
?2?
2__inference_custom_model_4_layer_call_fn_300119646
2__inference_custom_model_4_layer_call_fn_300119807
2__inference_custom_model_4_layer_call_fn_300120364
2__inference_custom_model_4_layer_call_fn_300120295?
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
?2?
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119392
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300120056
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300120226
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119484?
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
$__inference__wrapped_model_300118575?
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
bets?????????
?2?
+__inference_model_8_layer_call_fn_300118801
+__inference_model_8_layer_call_fn_300120474
+__inference_model_8_layer_call_fn_300118756
+__inference_model_8_layer_call_fn_300120461?
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
F__inference_model_8_layer_call_and_return_conditional_losses_300120448
F__inference_model_8_layer_call_and_return_conditional_losses_300120406
F__inference_model_8_layer_call_and_return_conditional_losses_300118678
F__inference_model_8_layer_call_and_return_conditional_losses_300118710?
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
+__inference_model_9_layer_call_fn_300120584
+__inference_model_9_layer_call_fn_300120571
+__inference_model_9_layer_call_fn_300118982
+__inference_model_9_layer_call_fn_300119027?
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
F__inference_model_9_layer_call_and_return_conditional_losses_300120516
F__inference_model_9_layer_call_and_return_conditional_losses_300118936
F__inference_model_9_layer_call_and_return_conditional_losses_300120558
F__inference_model_9_layer_call_and_return_conditional_losses_300118904?
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
,__inference_dense_36_layer_call_fn_300120604?
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
G__inference_dense_36_layer_call_and_return_conditional_losses_300120595?
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
,__inference_dense_37_layer_call_fn_300120624?
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
G__inference_dense_37_layer_call_and_return_conditional_losses_300120615?
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
,__inference_dense_39_layer_call_fn_300120643?
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
G__inference_dense_39_layer_call_and_return_conditional_losses_300120634?
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
,__inference_dense_38_layer_call_fn_300120663?
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
G__inference_dense_38_layer_call_and_return_conditional_losses_300120654?
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
,__inference_dense_40_layer_call_fn_300120682?
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
G__inference_dense_40_layer_call_and_return_conditional_losses_300120673?
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
,__inference_dense_41_layer_call_fn_300120701?
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
G__inference_dense_41_layer_call_and_return_conditional_losses_300120692?
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
,__inference_dense_42_layer_call_fn_300120720?
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
G__inference_dense_42_layer_call_and_return_conditional_losses_300120711?
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
,__inference_dense_43_layer_call_fn_300120739?
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
G__inference_dense_43_layer_call_and_return_conditional_losses_300120730?
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
/__inference_normalize_4_layer_call_fn_300120765?
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
J__inference_normalize_4_layer_call_and_return_conditional_losses_300120756?
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
,__inference_dense_44_layer_call_fn_300120784?
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
G__inference_dense_44_layer_call_and_return_conditional_losses_300120775?
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
'__inference_signature_wrapper_300119886betscards0cards1"?
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
-__inference_flatten_8_layer_call_fn_300120795?
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
H__inference_flatten_8_layer_call_and_return_conditional_losses_300120790?
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
0__inference_embedding_26_layer_call_fn_300120812?
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
K__inference_embedding_26_layer_call_and_return_conditional_losses_300120805?
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
0__inference_embedding_24_layer_call_fn_300120829?
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
K__inference_embedding_24_layer_call_and_return_conditional_losses_300120822?
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
0__inference_embedding_25_layer_call_fn_300120846?
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
K__inference_embedding_25_layer_call_and_return_conditional_losses_300120839?
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
-__inference_flatten_9_layer_call_fn_300120857?
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
H__inference_flatten_9_layer_call_and_return_conditional_losses_300120852?
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
0__inference_embedding_29_layer_call_fn_300120874?
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
K__inference_embedding_29_layer_call_and_return_conditional_losses_300120867?
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
0__inference_embedding_27_layer_call_fn_300120891?
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
K__inference_embedding_27_layer_call_and_return_conditional_losses_300120884?
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
0__inference_embedding_28_layer_call_fn_300120908?
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
K__inference_embedding_28_layer_call_and_return_conditional_losses_300120901?
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
$__inference__wrapped_model_300118575?.???????????LMYZST_`eflmst{|????{?x
q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????
? "3?0
.
dense_44"?
dense_44??????????
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119392?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????
p

 
? "%?"
?
0?????????
? ?
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300119484?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????
p 

 
? "%?"
?
0?????????
? ?
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300120056?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
M__inference_custom_model_4_layer_call_and_return_conditional_losses_300120226?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
2__inference_custom_model_4_layer_call_fn_300119646?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????
p

 
? "???????????
2__inference_custom_model_4_layer_call_fn_300119807?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????
p 

 
? "???????????
2__inference_custom_model_4_layer_call_fn_300120295?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????
p

 
? "???????????
2__inference_custom_model_4_layer_call_fn_300120364?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????
p 

 
? "???????????
G__inference_dense_36_layer_call_and_return_conditional_losses_300120595^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_36_layer_call_fn_300120604QLM0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_37_layer_call_and_return_conditional_losses_300120615^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_37_layer_call_fn_300120624QST0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_38_layer_call_and_return_conditional_losses_300120654^_`0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_38_layer_call_fn_300120663Q_`0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_39_layer_call_and_return_conditional_losses_300120634]YZ/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_dense_39_layer_call_fn_300120643PYZ/?,
%?"
 ?
inputs?????????
? "????????????
G__inference_dense_40_layer_call_and_return_conditional_losses_300120673^ef0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_40_layer_call_fn_300120682Qef0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_41_layer_call_and_return_conditional_losses_300120692^lm0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_41_layer_call_fn_300120701Qlm0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_42_layer_call_and_return_conditional_losses_300120711^st0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_42_layer_call_fn_300120720Qst0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_43_layer_call_and_return_conditional_losses_300120730^{|0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_43_layer_call_fn_300120739Q{|0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_44_layer_call_and_return_conditional_losses_300120775_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
,__inference_dense_44_layer_call_fn_300120784R??0?-
&?#
!?
inputs??????????
? "???????????
K__inference_embedding_24_layer_call_and_return_conditional_losses_300120822a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_24_layer_call_fn_300120829T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_25_layer_call_and_return_conditional_losses_300120839a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_25_layer_call_fn_300120846T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_26_layer_call_and_return_conditional_losses_300120805a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_26_layer_call_fn_300120812T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_27_layer_call_and_return_conditional_losses_300120884a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_27_layer_call_fn_300120891T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_28_layer_call_and_return_conditional_losses_300120901a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_28_layer_call_fn_300120908T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_29_layer_call_and_return_conditional_losses_300120867a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_29_layer_call_fn_300120874T?/?,
%?"
 ?
inputs?????????
? "????????????
H__inference_flatten_8_layer_call_and_return_conditional_losses_300120790X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_flatten_8_layer_call_fn_300120795K/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_flatten_9_layer_call_and_return_conditional_losses_300120852X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_flatten_9_layer_call_fn_300120857K/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_model_8_layer_call_and_return_conditional_losses_300118678l????8?5
.?+
!?
input_9?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_8_layer_call_and_return_conditional_losses_300118710l????8?5
.?+
!?
input_9?????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_8_layer_call_and_return_conditional_losses_300120406k????7?4
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
F__inference_model_8_layer_call_and_return_conditional_losses_300120448k????7?4
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
+__inference_model_8_layer_call_fn_300118756_????8?5
.?+
!?
input_9?????????
p

 
? "????????????
+__inference_model_8_layer_call_fn_300118801_????8?5
.?+
!?
input_9?????????
p 

 
? "????????????
+__inference_model_8_layer_call_fn_300120461^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
+__inference_model_8_layer_call_fn_300120474^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
F__inference_model_9_layer_call_and_return_conditional_losses_300118904m????9?6
/?,
"?
input_10?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_9_layer_call_and_return_conditional_losses_300118936m????9?6
/?,
"?
input_10?????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_9_layer_call_and_return_conditional_losses_300120516k????7?4
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
F__inference_model_9_layer_call_and_return_conditional_losses_300120558k????7?4
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
+__inference_model_9_layer_call_fn_300118982`????9?6
/?,
"?
input_10?????????
p

 
? "????????????
+__inference_model_9_layer_call_fn_300119027`????9?6
/?,
"?
input_10?????????
p 

 
? "????????????
+__inference_model_9_layer_call_fn_300120571^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
+__inference_model_9_layer_call_fn_300120584^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
J__inference_normalize_4_layer_call_and_return_conditional_losses_300120756[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? ?
/__inference_normalize_4_layer_call_fn_300120765N??+?(
!?
?
x??????????
? "????????????
'__inference_signature_wrapper_300119886?.???????????LMYZST_`eflmst{|???????
? 
???
&
bets?
bets?????????
*
cards0 ?
cards0?????????
*
cards1 ?
cards1?????????"3?0
.
dense_44"?
dense_44?????????